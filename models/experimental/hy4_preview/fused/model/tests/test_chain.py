# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint layers 0 to 5 chained on the 8x4 mesh (fused/model/program.py), free-running from the embedding, against
HF's decode golden of the 6-layer model (reference/hf_harness.py record_decode_golden: every layer's stages at the
decode positions, HF's own cache rows, the head readout, the logits and the greedy token).

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k accuracy
    pytest <this file> -x -k trace          (only after the untraced run has passed)

A step: 8 users = 8 decode positions of the golden (HY4_CHAIN_PROMPT=31, the default: positions 30 .. 61 in four steps
of 8; 1022: positions 1021 .. 1026, the golden tests/test_model_prefill_decode.py reports on, as users 0..5, users 6
and 7 repeating the first two). User u's cache holds, in every layer, HF's cache rows of positions 0 .. P_u (its own
row included: the append kernel is not built) packed by the prefill stack's ops into ONE packed-row storage of
8 users x 6 layers; the selection is every cached position on every layer (the contexts are shorter than the top-k:
what the reference indexer picks on layers 0, 1 and 5 too); rows are striped over the chips in blocks of BLOCK
positions so that every chip holds rows. The streams start as HF's expanded embedding of the 8 tokens.

accuracy   the 19 programs launched one by one, the host reading between them: per layer x against attn_in / mlp_in,
           F against attn_out / mlp_out (PCC and gain), the streams after each sublayer, the router's picks against
           HF's and against a host router on the device's own x; then head readout, final norm and LM head
           (reference/ihc.py, reference/lm_head.py on the host, from the device's streams): logits and the greedy token
           against HF's. Aggregates over all rows and layers at the end (lines HY4_MK11 ... TABLE).
trace      the same step as ONE trace: final streams against the untraced step's and HF's, the canary, then latency:
           the whole step, the attention programs alone, the MLP programs alone, and prefixes of 1 .. 6 layers.
"""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.layer_attn import program as A
from models.experimental.hy4_preview.fused.layer_common import transport as tp
from models.experimental.hy4_preview.fused.model.program import Stack, is_mock
from models.experimental.hy4_preview.fused.moe import layout as ML
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections import prepare as PP
from models.experimental.hy4_preview.fused.sparse_mla import layout as SL
from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.attention import AttentionConfig, rope_cos_sin, rope_pairing
from models.experimental.hy4_preview.reference.ihc import IHCConfig, IHCHead
from models.experimental.hy4_preview.reference.lm_head import Hy4LMHead, LMHeadConfig
from models.experimental.hy4_preview.tests.common import hy4_tensors, pcc
from models.experimental.hy4_preview.tt import frames
from models.experimental.hy4_preview.tt import prefill_bridge as pb

MESH = (8, 4)
CHIPS = 32
LAYERS = int(os.environ.get("HY4_CHAIN_LAYERS", "6"))  # 6, 12, 24, 48, 78: the depths with a decode golden
TRACE_REGION = 128 << 20
FIRST_WAIT = 100_000_000
PROMPT = int(os.environ.get("HY4_CHAIN_PROMPT", "31"))
BLOCK = 1 if PROMPT < 512 else 4
MOE_FIDELITY = os.environ.get("HY4_MOE_FIDELITY", "p02")
LATENT_SCALE = float(os.environ["HY4_LATENT_SCALE"]) if os.environ.get("HY4_LATENT_SCALE") else None
JOB = os.environ.get("HY4_CHAIN_TRACE_JOB", "A")  # A: capture and one replay, then stop; B: the replays and the latency
# (the dumps of the two RoPE pairings, reference/attention.py rope_pairing, are different models' numbers)
PAIRING = rope_pairing()
DUMP = (
    f"generated/hy4_preview/logs/mk11_chain{'' if LAYERS == 6 else f'_L{LAYERS}'}_P{PROMPT}_{MOE_FIDELITY}"
    f"{'' if LATENT_SCALE is None else '_scaled'}{'' if PAIRING == 'half_split' else '_' + PAIRING}.pt"
)
GOLDEN_STEPS = {31: 32, 33: 32, 1000: 6, 1022: 6, 3000: 6, 5118: 6}
# A MIXED step: the 8 rows from different goldens, "prompt:step" per row (Chain(mix=...)), e.g.
# HY4_CHAIN_MIX=31:0,31:1,1022:2,1022:3,1000:4,1000:5,3000:0,3000:1 : row u is the row that step has in its own golden's
# batch 0 (same user slot, position, cache rows, striping, selection and token), beside rows of other conversations
MIX = os.environ.get("HY4_CHAIN_MIX")
galaxy = pytest.mark.parametrize(
    "device_params, mesh_device",
    [(dict(pb.device_params(), trace_region_size=TRACE_REGION), MESH)],
    indirect=True,
    ids=["8x4"],
)


def rope_tiles(positions, acfg):
    """(cos, sin) as Stack.set_rope takes them for the rows' positions. THE one place of this piece's tests that fixes
    the pairing of the RoPE dimensions: HF's tables (reference/attention.py rope_cos_sin: half-split) through
    tt/frames.py rope_tables (to the interleaved order the device's rows are stored in). The golden is the other
    parameter: hf_harness.find_decode_golden under HY4_GOLDEN_DIR."""
    cos, sin = rope_cos_sin(torch.tensor(list(positions)), acfg)
    return frames.rope_tables(cos, sin)


def gain(ref, dev):
    ref, dev = ref.double().flatten(), dev.double().flatten()
    return float((ref * dev).sum() / (ref * ref).sum())


def block_of(prompt):
    """Positions per chip in a row's striping: every chip holds rows of a short context too."""
    return 1 if prompt < 512 else 4


def chip_of(p, block=BLOCK):
    return (p // block) % CHIPS


def row_of(p, block=BLOCK):
    return (p // (block * CHIPS)) * block + p % block


def parse_mix(text):
    """ "31:0,1022:2,..." -> [(prompt, step), ...] (8 rows)."""
    rows = [tuple(int(v) for v in item.split(":")) for item in text.split(",")]
    assert len(rows) == 8, f"a mixed step has 8 rows: {text}"
    return rows


def step_indices(batch):
    """The golden's step indices of the 8 users of batch ``batch``."""
    golden_steps = hf_harness.find_decode_golden(LAYERS, PROMPT)["meta"]["steps"] if batch else GOLDEN_STEPS[PROMPT]
    if golden_steps >= 8 * (batch + 1):
        return list(range(8 * batch, 8 * batch + 8))
    assert batch == 0
    return [j % golden_steps for j in range(8)]


def site(layer, name):
    prefix = f"model.layers.{layer}.{name}.hc_pre."
    return tuple(hy4_tensors([prefix + "hc_fn", prefix + "hc_base", prefix + "hc_scale"]))


class Chain:
    """The six layers' weights on the device and the golden. ``set_batch`` loads a step's 8 users."""

    def __init__(
        self,
        mesh_device,
        layers=LAYERS,
        timing=False,
        wait_limit=FIRST_WAIT,
        canary=True,
        load=True,
        attn_kwargs=None,
        head_site=False,
        mix=None,
    ):
        """load=False: only the Stack and both MlpLayers (every L1 tensor of the step) are made; ``load_layers()``
        brings the weights later (a caller that runs something else on the device in between).
        mix: [(prompt, step)] x 8 (default: HY4_CHAIN_MIX): the rows of batch 0 come from DIFFERENT goldens, each row
        with its own cache rows, selection and token; ``g`` and ``N`` are then None (a row's golden: ``rows``; use
        gold / gold_top / tokens)."""
        self.mesh, self.n = mesh_device, layers
        self.mix = mix if mix is not None else (parse_mix(MIX) if MIX else None)
        if self.mix:
            self.goldens = {P: hf_harness.find_decode_golden(LAYERS, P) for P in sorted({P for P, _ in self.mix})}
            missing = [P for P, g in self.goldens.items() if g is None]
            if missing:
                pytest.skip(f"no decode golden of {missing} tokens under {hf_harness.GOLDEN_ROOT}")
            for P, j in self.mix:
                assert 0 <= j < self.goldens[P]["meta"]["steps"], f"the golden of {P} tokens has no step {j}"
            self.g, self.N = None, None
            all_goldens = list(self.goldens.values())
            self.label = "MIX " + ",".join(f"{P}:{j}" for P, j in self.mix)
        else:
            self.g = g = hf_harness.find_decode_golden(LAYERS, PROMPT)
            if g is None:
                pytest.skip(f"no decode golden of a {PROMPT}-token prompt under {hf_harness.GOLDEN_ROOT}")
            self.N = g["meta"]["prompt_len"]
            all_goldens = [g]
            self.label = f"P{PROMPT}"
        self.rows = None  # per row (its golden, its step, its striping block): set_batch
        # a context above the top-k: the reference's indexer drops visible keys, and each layer reads by ITS picks
        # (the golden's topk_indices: computed on the full-indexer layers, repeated on the shared layers behind them)
        self.picked = any(sum(layer["dsa_dropped_rows"]) for g_ in all_goldens for layer in g_["layers"])
        self.sels = []
        self.acfg = AttentionConfig()
        if (
            LATENT_SCALE is not None
        ):  # AttnLayer(latent_scale=): one scalar on the stored rows' latent part, in the weights
            attn_kwargs = dict(attn_kwargs or {}, latent_scale=LATENT_SCALE)
        self.st = st = Stack(
            mesh_device,
            moe_fidelity=MOE_FIDELITY,
            timing=timing,
            wait_limit=wait_limit,
            canary=canary,
            attn_kwargs=attn_kwargs,
        )
        # head_site: the site table gets one more entry behind the layers' 2 N: the model's hc_head with the final
        # norm's weight as its gamma (ends_fused.head_site; index 2 N), for the fused ends' HEAD program
        self.head_site = head_site
        self.router = {}
        self.kv = self.sel = None
        self.J = None
        if load:
            self.load_layers()
        else:
            for dense in (True, False)[: 2 if layers > 1 else 1]:
                st._mlp(dense)

    def _rows(self, batch):
        """[(golden, step, striping block)] of the 8 rows of ``batch``."""
        if self.mix:
            assert batch == 0, "a mixed step is batch 0"
            return [(self.goldens[P], j, block_of(P)) for P, j in self.mix]
        return [(self.g, j, BLOCK) for j in step_indices(batch)]

    def load_layers(self):
        st, layers, mesh_device = self.st, self.n, self.mesh
        t0 = time.perf_counter()
        prefix = "model.layers.{}."
        gammas = hy4_tensors(
            [
                prefix.format(n) + name
                for n in range(layers)
                for name in ("input_layernorm.weight", "post_attention_layernorm.weight")
            ]
        )
        sites = [s for n in range(layers) for s in (site(n, "hc_attn_layer"), site(n, "hc_mlp_layer"))]
        if self.head_site:
            from models.experimental.hy4_preview.fused.model import ends_fused

            fn, base, scale, norm_w = hy4_tensors(
                [
                    "model.hc_head.hc_head_fn",
                    "model.hc_head.hc_head_base",
                    "model.hc_head.hc_head_scale",
                    "model.norm.weight",
                ]
            )
            sites, gammas = sites + [ends_fused.head_site(fn, base, scale)], list(gammas) + [norm_w]
        st.set_sites(sites, gammas)
        for n in range(layers):
            attention, _, _ = PP.checkpoint_layer(n)
            lw = L.layer_weights(attention, None, None)  # the shared-indexer form: the selection comes from the host
            sink = attention["learnable_sink_param"].float().flatten()
            dense = n == 0
            rw = rb = None
            if not dense:
                rw, rb = hy4_tensors(
                    [prefix.format(n) + "mlp.gate.weight", prefix.format(n) + "mlp.gate.e_score_correction_bias"]
                )
                self.router[n] = (rw, rb)
            st.add_layer(n, lw, sink, 2 * n, 2 * n + 1, cache_layer=n, dense=dense, router_weight=rw, router_bias=rb)
            if n < 6 or n % 6 == 5:
                logger.info(
                    f"HY4_MK11 chain: layer {n} loaded ({time.perf_counter() - t0:.0f} s), L1 tensors {st.l1_bytes()} B per core"
                )
        dram = pb.dram(mesh_device)
        logger.info(
            f"HY4_MK11 chain: {layers} layers loaded in {time.perf_counter() - t0:.0f} s; L1 tensors {st.l1_bytes()} B per core; DRAM "
            f"{ {k: round(v, 3) if isinstance(v, float) else v for k, v in dram.items()} }; program-cache entries {mesh_device.num_program_cache_entries()}"
        )
        # (the storage is sized for the longest row: a row's golden holds its positions up to N + steps - 2)
        spans = [(self.goldens[P], block_of(P)) for P, _ in self.mix] if self.mix else [(self.g, BLOCK)]
        max_rows = max(row_of(g["meta"]["prompt_len"] + g["meta"]["steps"] - 2, block) + 1 for g, block in spans)
        self.cfg = SL.SparseMlaConfig(rows_per_slot=32 * -(-max_rows // 32), num_layers=layers)

    def set_batch(self, batch, steps=1):
        """The cache rows, the selection, the RoPE tables and the embedding streams of the 8 users of ``batch``.
        steps > 1: the storage also holds HF's rows of the next steps - 1 positions of every user, and ``step_inputs``
        holds, per following step, its selection table and RoPE tensor (use_step puts them in place)."""
        cfg, mesh = self.cfg, self.mesh
        self.rows = batch_rows = self._rows(batch)
        self.J = J = [j for _, j, _ in batch_rows]
        self.positions = [g["meta"]["prompt_len"] - 1 + j for g, j, _ in batch_rows]
        blocks = [block for _, _, block in batch_rows]
        users, n = len(J), self.n
        lat = torch.zeros(CHIPS, users * n, cfg.rows_per_slot, cfg.latent)
        rope = torch.zeros(CHIPS, users * n, cfg.rows_per_slot, cfg.rope)
        stored = {
            (c, u): [p for p in range(p_u + steps) if chip_of(p, blocks[u]) == c]
            for c in range(CHIPS)
            for u, p_u in enumerate(self.positions)
        }
        mine = {
            (c, u): [p for p in range(p_u + 1) if chip_of(p, blocks[u]) == c]
            for c in range(CHIPS)
            for u, p_u in enumerate(self.positions)
        }
        for layer in range(n):
            held_of = {}  # (each row's cache rows from ITS golden)
            for (c, u), ps in stored.items():
                g = batch_rows[u][0]
                if id(g) not in held_of:
                    held_of[id(g)] = frames.tail_to_device(
                        g["cache"][layer]["latent"].float(), self.acfg.qk_rope_head_dim
                    )
                held = held_of[id(g)]
                rows = torch.tensor([row_of(p, blocks[u]) for p in ps], dtype=torch.long)
                lat[c, cfg.slot(u, layer), rows] = held[ps, : cfg.latent]
                rope[c, cfg.slot(u, layer), rows] = held[ps, cfg.latent :]
        tables = [
            SL.selection_table(
                cfg,
                [torch.tensor([row_of(p, blocks[u]) for p in mine[(c, u)]], dtype=torch.long) for u in range(users)],
            )
            for c in range(CHIPS)
        ]
        mesh_bf16 = lambda t: ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        tt_lat = mesh_bf16(lat.reshape(CHIPS * users * n, 1, cfg.rows_per_slot, cfg.latent))
        tt_rope = mesh_bf16(rope.reshape(CHIPS * users * n, 1, cfg.rows_per_slot, cfg.rope))
        fp8, scales = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(tt_lat, round_scale_to_power_of_two=True)
        packed = ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache(fp8, scales, tt_rope)
        old = (self.kv, self.sel)
        self.kv = ttnn.to_memory_config(packed, hs.kv_memory_config(mesh, cfg))
        for t in (tt_lat, tt_rope, fp8, scales, packed):
            ttnn.deallocate(t)
        shards = [ttnn.from_torch(t[None, None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT) for t in tables]
        self.sel = ttnn.to_device(
            ttnn.from_host_shards(shards, mesh.shape), mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.st.set_cache(cfg, self.kv, self.sel)
        for t in list(old) + self.sels:
            if t is not None:
                ttnn.deallocate(t)
        self.sels = []
        if self.picked:
            # per layer: every row's picks (positions up to its own, as the reference's indexer chose them) as each
            # chip's stripe rows; the picks of a shared layer are those of the full layer before it (one table)
            assert steps == 1, "per-layer selections: one step"
            self.pick_counts = []
            for layer in range(n):
                picks = [g["layers"][layer]["topk_indices"][j] for g, j, _ in batch_rows]  # per row [top-k or fewer]
                if layer and all(
                    torch.equal(picks[u], g["layers"][layer - 1]["topk_indices"][j])
                    for u, (g, j, _) in enumerate(batch_rows)
                ):
                    self.st.layers[layer].sel = self.st.layers[layer - 1].sel
                    self.pick_counts.append(self.pick_counts[-1])
                    continue
                chosen = [
                    sorted(set(int(v) for v in picks[u].tolist() if 0 <= int(v) <= p_u))
                    for u, p_u in enumerate(self.positions)
                ]
                # (a row below the top-k reads every position up to its own; one above it exactly top-k)
                assert all(len(chosen[u]) == min(p_u + 1, 2048) for u, p_u in enumerate(self.positions)), [
                    len(c) for c in chosen
                ]
                tabs = [
                    SL.selection_table(
                        cfg,
                        [
                            torch.tensor(
                                [row_of(q, blocks[u]) for q in chosen[u] if chip_of(q, blocks[u]) == c],
                                dtype=torch.long,
                            )
                            for u in range(users)
                        ],
                    )
                    for c in range(CHIPS)
                ]
                sh = [ttnn.from_torch(t[None, None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT) for t in tabs]
                self.sels.append(
                    ttnn.to_device(ttnn.from_host_shards(sh, mesh.shape), mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                )
                self.st.layers[layer].sel = self.sels[-1]
                self.pick_counts.append([len(c) for c in chosen])
            logger.info(
                f"HY4_MK11 chain: a context above the top-k: per-layer selections from the golden's picks (the reference's indexer): "
                f"{len(self.sels)} tables for {n} layers; positions picked per row, per layer {self.pick_counts}; visible positions "
                f"per row {[p + 1 for p in self.positions]}"
            )
        self.st.set_rope(*rope_tiles(self.positions, self.acfg))
        a = self.st.a
        self.step_inputs = [(self.sel, a.rope)]
        for k in range(1, steps):  # the following steps: every user one position on
            seen = {
                (c, u): [p for p in range(p_u + k + 1) if chip_of(p, blocks[u]) == c]
                for c in range(CHIPS)
                for u, p_u in enumerate(self.positions)
            }
            tabs = [
                SL.selection_table(
                    cfg,
                    [
                        torch.tensor([row_of(p, blocks[u]) for p in seen[(c, u)]], dtype=torch.long)
                        for u in range(users)
                    ],
                )
                for c in range(CHIPS)
            ]
            sh = [ttnn.from_torch(t[None, None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT) for t in tabs]
            sel = ttnn.to_device(ttnn.from_host_shards(sh, mesh.shape), mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            first = a.rope
            a.rope = tp.core_buffer(mesh, [A.QUERY_HUB], 2 * A.ROW_TILE)
            self.st.set_rope(*rope_tiles([p + k for p in self.positions], self.acfg))
            self.step_inputs.append((sel, a.rope))
            a.rope = first
        self.rows_per_chip = [len(v) for v in mine.values()]
        self.reset_streams()
        ttnn.synchronize_device(mesh)

    def stored_rows(self):
        """{(layer, user): [P_u + 1, 576]}: the cache rows as the chips hold them (the packed storage read back through
        the prefill stack's unpack), in position order, HF frame."""
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
            MlaKvCacheGeometry,
            reconstruct_scaled_fp8_kv_cache,
        )

        cfg = self.cfg
        physical = ttnn.to_torch(self.kv, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        geometry = MlaKvCacheGeometry(latent_dim=cfg.latent, rope_dim=cfg.rope)
        rows = (
            reconstruct_scaled_fp8_kv_cache(physical, geometry)[:, 0]
            .float()
            .reshape(CHIPS, -1, cfg.rows_per_slot, cfg.width)
        )
        out = {}
        for layer in range(self.n):
            for u, p_u in enumerate(self.positions):
                block = self.rows[u][2]
                held = torch.stack(
                    [rows[chip_of(p, block), cfg.slot(u, layer), row_of(p, block)] for p in range(p_u + 1)]
                )
                out[(layer, u)] = frames.tail_from_device(held, self.acfg.qk_rope_head_dim).to(torch.float32)
        return out

    def use_step(self, k):
        """Put step k's selection table and RoPE tensor in place (before that step's descriptors are built)."""
        a = self.st.a
        a.sel, a.rope = self.step_inputs[k]
        # (Stack._use puts the Stack's own table back before every layer's descriptors: without this line step k ran
        # on step 0's selection, i.e. its rows did not read their own position: PLAN.md section 48)
        self.st._sel = a.sel
        a._programs = {}

    def reset_streams(self):
        self.st.set_streams(self.gold(0, "streams_in"))

    def row_goldens(self):
        """[(golden, step)] per row. One golden: the rows are its steps ``J`` (a caller may have set J itself, as
        tests/test_real.py prepare_step does); a mixed step: as set_batch left them."""
        if self.mix:
            return [(g, j) for g, j, _ in self.rows]
        return [(self.g, j) for j in self.J]

    def gold(self, layer, stage, k=0):
        """[8, ...]: each row's golden's ``stage`` of ``layer`` at the row's step + k."""
        return torch.stack([g["layers"][layer][stage][j + k] for g, j in self.row_goldens()]).float()

    def gold_top(self, name, k=0):
        """[8, ...]: each row's golden's top-level tensor ("logits", "cached_logits_argmax", "hc_head") at its step + k."""
        return torch.stack([g[name][j + k] for g, j in self.row_goldens()])

    def tokens(self, k=0):
        """The token each row is fed at its step + k (its golden's input id at its position)."""
        return [int(g["input_ids"][g["meta"]["prompt_len"] - 1 + j + k]) for g, j in self.row_goldens()]

    def gold_unions(self):
        """Per MoE layer: the number of distinct experts the 8 rows' goldens pick (the reference router's)."""
        out = []
        rows = self.row_goldens()
        for layer in range(self.n):
            if "router_indices" in rows[0][0]["layers"][layer]:
                out.append(
                    len({int(v) for g, j in rows for v in g["layers"][layer]["router_indices"][j].flatten().tolist()})
                )
        return out

    def free(self):
        for t in [self.kv, self.sel] + self.sels:
            if t is not None:
                ttnn.deallocate(t)


def head_reference():
    """(IHCHead, Hy4LMHead in HF's dtypes) with the checkpoint's parameters, on the host."""
    fn, base, scale, norm_w, head_w = hy4_tensors(
        [
            "model.hc_head.hc_head_fn",
            "model.hc_head.hc_head_base",
            "model.hc_head.hc_head_scale",
            "model.norm.weight",
            "lm_head.weight",
        ]
    )
    head = IHCHead(IHCConfig()).eval()
    lm = Hy4LMHead(LMHeadConfig(), init=False).eval()
    with torch.no_grad():
        head.fn.copy_(fn), head.base.copy_(base), head.scale.copy_(scale.reshape(head.scale.shape))
        lm.norm_weight.copy_(norm_w)
        lm.weight = torch.nn.Parameter(head_w, requires_grad=False)
    return head, lm.as_hf_bf16()


def host_logits(head, lm, streams):
    """HF's tail on the device's streams [8, 4, 6144]: (head readout, logits), BF16 hidden state as HF holds it."""
    with torch.no_grad():
        h = head(streams.to(torch.bfloat16).unsqueeze(0))[0]
        return h.float(), lm(h).float()


def run_step_untraced(chain, label, records, after_program=None):
    """One step, program by program, the host reading between them. Appends one record per (layer, quantity).
    after_program(kind, layer): called after each program has finished, before this function's own reads."""
    st, a, mesh = chain.st, chain.st.a, chain.mesh
    programs = st.step_programs(audit=True)
    emb = chain.gold(0, "streams_in")
    out = {}
    for name, io, desc in programs:
        Stack.launch([(name, io, desc)])
        ttnn.synchronize_device(mesh)
        kind = name.split("[")[0]
        layer = int(name.split("[")[1][:-1]) if "[" in name else chain.n
        if after_program is not None:
            after_program(kind, layer)
        if kind == "p1":
            status = a.status(p1_only=True)
            # (P2's relays take the X3 arrivals: they stay counted here, and must not be reset)
            assert not status, f"{label} {name}: words not at rest: {status}"
            root = a.read_root()
            out[(layer, "attn_in")] = root["x"]
            if layer > 0:
                out[(layer - 1, "streams_out")] = st.read_streams()
        elif kind == "p2":
            status = a.status()
            assert not status, f"{label} {name}: words not at rest: {status}"
            out[(layer, "attn_out")] = a.read_root()["f"]
        elif kind == "mlp":
            status = st.layers[layer].mlp.status()
            assert not status, f"{label} {name}: words not at rest: {status}"
            root = a.read_root()
            out[(layer, "streams_mid")] = st.read_streams()
            out[(layer, "mlp_in")] = root["x"]
            out[(layer, "mlp_out")] = root["f"]
            if layer in chain.router:
                out[(layer, "picks")] = st.read_mlp_chip(st.layers[layer], st.topo.root)
        else:
            out[(chain.n - 1, "streams_out")] = st.read_streams()
        # stop at the first program that leaves anything not finite (its layer is in the message)
        for key, value in out.items():
            if key[0] in (layer, layer - 1) and isinstance(value, torch.Tensor):
                assert bool(torch.isfinite(value).all()), f"{label} {name}: {key[1]} of layer {key[0]} is not finite"
    status = st.status()
    assert not status, f"{label}: words not at rest after the step: {status}"
    for layer in range(chain.n):
        g = lambda stage: chain.gold(layer, stage)
        rec = {"layer": layer, "label": label}
        for stage in ("attn_in", "attn_out", "streams_mid", "mlp_in", "mlp_out", "streams_out"):
            rec[stage] = (pcc(g(stage), out[(layer, stage)]), gain(g(stage), out[(layer, stage)]))
        rec["rows_pcc"] = {
            stage: [pcc(g(stage)[r], out[(layer, stage)][r]) for r in range(8)]
            for stage in ("attn_out", "mlp_out", "streams_out")
        }
        upd_g, upd_d = g("streams_out") - emb, out[(layer, "streams_out")] - emb
        rec["update"] = (pcc(upd_g, upd_d), gain(upd_g, upd_d))
        own_g, own_d = g("streams_out") - g("streams_in"), out[(layer, "streams_out")] - (
            out[(layer - 1, "streams_out")] if layer else emb
        )
        rec["own_update"] = (pcc(own_g, own_d), gain(own_g, own_d))
        text = ""
        if layer in chain.router:
            picks = out[(layer, "picks")]
            ids = picks["ids"].long()
            hf_ids = chain.g["layers"][layer]["router_indices"][chain.J]
            rw, rb = chain.router[layer]
            r_scores, r_ids, _ = ML.route_golden(out[(layer, "mlp_in")], rw, rb)
            biased = r_scores + rb.float()
            cut = biased.topk(8, dim=-1).values[:, -1]
            rec["rows_hf"] = [set(ids[r].tolist()) == set(hf_ids[r].tolist()) for r in range(8)]
            rec["shared_hf"] = sum(len(set(ids[r].tolist()) & set(hf_ids[r].tolist())) for r in range(8))
            rec["rows_host"] = sum(set(ids[r].tolist()) == set(r_ids[r].tolist()) for r in range(8))
            rec["below"] = float((cut - biased.gather(1, ids).min(dim=-1).values).max())
            rec["margin"] = float((cut - biased.topk(9, dim=-1).values[:, -1]).min())
            rec["union"] = len(set(ids.flatten().tolist()))
            same = [r for r in range(8) if rec["rows_hf"][r]]
            rec["mlp_out_same_picks"] = pcc(g("mlp_out")[same], out[(layer, "mlp_out")][same]) if same else None
            text = (
                f"; picks: {sum(rec['rows_hf'])}/8 rows HF's 8 experts ({rec['shared_hf']}/64 shared), {rec['rows_host']}/8 rows a host "
                f"router's on the device's x (worst pick {rec['below']:.1e} below its cut, smallest 8th-9th margin {rec['margin']:.1e}), "
                f"union {rec['union']}; mlp_out on the rows with HF's picks pcc "
                f"{rec['mlp_out_same_picks'] if rec['mlp_out_same_picks'] is None else round(rec['mlp_out_same_picks'], 5)}"
            )
        logger.info(
            f"HY4_MK11 {label} layer {layer} vs HF (free-running): attn_in pcc={rec['attn_in'][0]:.6f}; attn_out pcc="
            f"{rec['attn_out'][0]:.5f} gain={rec['attn_out'][1]:.4f}; streams_mid pcc={rec['streams_mid'][0]:.6f}; mlp_in pcc="
            f"{rec['mlp_in'][0]:.6f}; mlp_out pcc={rec['mlp_out'][0]:.5f} gain={rec['mlp_out'][1]:.4f}; streams_out pcc="
            f"{rec['streams_out'][0]:.6f} gain={rec['streams_out'][1]:.4f}; update since the embedding pcc={rec['update'][0]:.5f} gain={rec['update'][1]:.4f}; the "
            f"layer's own update pcc={rec['own_update'][0]:.5f} gain={rec['own_update'][1]:.4f}{text}"
        )
        records.append(rec)
    return out


def check_logits(chain, head, lm, streams, label):
    g, J = chain.g, chain.J
    h, logits = host_logits(head, lm, streams)
    gh, gl = g["hc_head"][J].float(), g["logits"][J].float()
    want = g["cached_logits_argmax"][J]
    got = logits.argmax(-1)
    _, ref_logits = host_logits(head, lm, chain.gold(chain.n - 1, "streams_out"))
    top5 = [int(want[r]) in logits[r].topk(5).indices.tolist() for r in range(8)]
    margins = [float(gl[r].topk(2).values[0] - gl[r].topk(2).values[1]) for r in range(8)]
    rows = [pcc(gl[r], logits[r]) for r in range(8)]
    res = {
        "readout": pcc(gh, h),
        "logits": pcc(gl, logits),
        "logits_min_row": min(rows),
        "tokens": int((got == want).sum()),
        "top5": sum(top5),
        "rows": len(J),
        "wrong": [(int(J[r]), int(want[r]), int(got[r]), round(margins[r], 3)) for r in range(8) if got[r] != want[r]],
    }
    logger.info(
        f"HY4_MK11 {label} head (reference tail on the device's streams): readout pcc={res['readout']:.6f}; logits pcc="
        f"{res['logits']:.6f} (per row min {min(rows):.6f}); greedy token equal to HF's on {res['tokens']} of 8 rows (HF's token in "
        f"the device's top 5 on {res['top5']}); differing rows (step, HF, device, HF's top-1 margin in logits) {res['wrong']}; the "
        f"reference tail on HF's own streams gives HF's logits at pcc {pcc(gl, ref_logits):.6f}"
    )
    return res


def table(records, heads, label):
    """Aggregates over all rows and layers."""
    stages = ("attn_in", "attn_out", "streams_mid", "mlp_in", "mlp_out", "streams_out", "update", "own_update")
    for layer in sorted({r["layer"] for r in records}):
        rows = [r for r in records if r["layer"] == layer]
        cell = lambda s: f"{min(r[s][0] for r in rows):.5f}/{statistics.mean(r[s][1] for r in rows):.4f}"
        picks = ""
        if "rows_hf" in rows[0]:
            picks = (
                f" | picks: rows with HF's 8 experts {sum(sum(r['rows_hf']) for r in rows)}/{8 * len(rows)}, picks shared "
                f"{sum(r['shared_hf'] for r in rows)}/{64 * len(rows)}, rows with a host router's picks on the device's x "
                f"{sum(r['rows_host'] for r in rows)}/{8 * len(rows)}, unions {[r['union'] for r in rows]}"
            )
        logger.info(
            f"HY4_MK11 {label} TABLE layer {layer} (min pcc / mean gain over {len(rows)} steps): "
            + " ".join(f"{s}={cell(s)}" for s in stages)
            + picks
        )
    layers = sorted({r["layer"] for r in records})
    if len(layers) > 6:  # the depth trend, per 6 layers: the streams after the last layer of each block
        for last in [n for n in layers if n % 6 == 5 or n == layers[-1]]:
            rows = [r for r in records if r["layer"] == last]
            block = [r for r in records if last - 5 <= r["layer"] <= last and "rows_hf" in r]
            logger.info(
                f"HY4_MK11 {label} DEPTH after layer {last}: streams pcc min {min(r['streams_out'][0] for r in rows):.5f} mean "
                f"{statistics.mean(r['streams_out'][0] for r in rows):.5f}, gain {statistics.mean(r['streams_out'][1] for r in rows):.4f}; "
                f"update since the embedding pcc {statistics.mean(r['update'][0] for r in rows):.5f} gain "
                f"{statistics.mean(r['update'][1] for r in rows):.4f}; in layers {max(last - 5, 0)}..{last}: picks shared with HF "
                f"{sum(r['shared_hf'] for r in block)}/{64 * len(block)} ({100 * sum(r['shared_hf'] for r in block) / max(1, 64 * len(block)):.1f} %), "
                f"rows with HF's 8 experts {sum(sum(r['rows_hf']) for r in block)}/{8 * len(block)}, attn_out gain "
                f"{statistics.mean(r['attn_out'][1] for r in records if last - 5 <= r['layer'] <= last):.4f}, mlp_out pcc mean "
                f"{statistics.mean(r['mlp_out'][0] for r in records if last - 5 <= r['layer'] <= last):.4f}"
            )
    moe = [r for r in records if "rows_hf" in r]
    if moe:
        logger.info(
            f"HY4_MK11 {label} TABLE picks over {8 * len(moe)} (row, layer) pairs: rows with HF's 8 experts "
            f"{sum(sum(r['rows_hf']) for r in moe)} ({100 * sum(sum(r['rows_hf']) for r in moe) / (8 * len(moe)):.1f} %), picks shared with HF "
            f"{sum(r['shared_hf'] for r in moe)}/{64 * len(moe)} ({100 * sum(r['shared_hf'] for r in moe) / (64 * len(moe)):.2f} %), rows with a host "
            f"router's picks on the device's own x {sum(r['rows_host'] for r in moe)}/{8 * len(moe)}; worst device pick "
            f"{max(r['below'] for r in moe):.1e} below a host router's cut"
        )
    if heads:
        logger.info(
            f"HY4_MK11 {label} TABLE head over {sum(h['rows'] for h in heads)} rows: readout pcc min {min(h['readout'] for h in heads):.6f}; "
            f"logits pcc min {min(h['logits'] for h in heads):.6f} (per row min {min(h['logits_min_row'] for h in heads):.6f}); greedy token "
            f"equal to HF's on {sum(h['tokens'] for h in heads)} of {sum(h['rows'] for h in heads)} rows, HF's token in the device's top 5 on "
            f"{sum(h['top5'] for h in heads)}; differing {[w for h in heads for w in h['wrong']]}"
        )


@galaxy
def test_accuracy(mesh_device):
    chain = Chain(mesh_device)
    st = chain.st
    batches = int(os.environ.get("HY4_CHAIN_BATCHES", "4" if PROMPT in (31, 33) else "1"))
    try:
        if is_mock():
            chain.set_batch(0)
            Stack.launch(st.step_programs(audit=True))
            return
        head, lm = head_reference()
        records, heads, dump = [], [], []
        for batch in range(batches):
            chain.set_batch(batch)
            label = (
                f"chain P{PROMPT} [{MOE_FIDELITY}] step {batch} (positions {chain.positions[0]}..{chain.positions[-1]})"
            )
            if batch == 0:
                report = st.l1_report(st.layers[-1])
                logger.info(
                    f"HY4_MK11 chain: L1 tensors {report['tensors']} of {report['core']} B per core, largest free block "
                    f"{report['largest_free']}, largest CB set of P2 {report[('p2', (0, 0))][0]}; rows per chip and user "
                    f"{min(chain.rows_per_chip)} to {max(chain.rows_per_chip)}; program-cache entries {mesh_device.num_program_cache_entries()}"
                )
            stored = chain.stored_rows()
            out = run_step_untraced(chain, label, records)
            heads.append(check_logits(chain, head, lm, out[(chain.n - 1, "streams_out")], label))
            dump.append(
                {
                    "J": list(chain.J),
                    "positions": list(chain.positions),
                    "stored": stored,
                    "out": {
                        k: (
                            v
                            if isinstance(v, torch.Tensor)
                            else {kk: vv for kk, vv in v.items() if kk in ("ids", "weights")}
                        )
                        for k, v in out.items()
                    },
                }
            )
            canary = st.canary_summary()
            logger.info(f"HY4_MK11 {label} canary on the 7 heads, totals so far: {canary}")
            for kind, v in canary.items():
                assert (
                    v["found"] == 0 and v["timeouts"] == 0 and v["checks"] == v["runs"]
                ), f"{label}: canary {kind}: {v}"
        logger.info(
            f"HY4_MK11 chain: program-cache entries after {batches} steps {mesh_device.num_program_cache_entries()}"
        )
        programs = st.step_programs()
        chain.reset_streams()
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        Stack.launch(programs)
        ttnn.synchronize_device(mesh_device)
        took = time.perf_counter() - start
        status = st.status()
        logger.info(
            f"HY4_MK11 chain: the {len(programs)} launches of a step of {chain.n} layers UNTRACED, back to back, one synchronize: "
            f"{1e3 * took:.1f} ms ({1e6 * took / len(programs):.0f} us per launch); words {status or 'at rest'}"
        )
        assert not status, f"words not at rest after the untimed step: {status}"
        table(records, heads, f"chain P{PROMPT} [{MOE_FIDELITY}] untraced")
        torch.save({"prompt": PROMPT, "fidelity": MOE_FIDELITY, "block": BLOCK, "steps": dump}, DUMP)
        logger.info(
            f"HY4_MK11 chain: the device's stages and the stored rows of {len(dump)} steps saved to {DUMP} (test_control reads it)"
        )
    finally:
        chain.free()


def full_status(chain) -> dict:
    """Every status of every piece after a step: both halves' words and every layer's MoE status (Stack.status), the
    projection groups' status pages and sparse_mla's status pages on every chip. Empty when nothing was left."""
    from models.experimental.hy4_preview.fused.sparse_mla.op import STATUS_MAGIC

    st, a = chain.st, chain.st.a
    out = dict(st.status())
    out_pages = ttnn.get_device_tensors(a.out)
    for coord in st.topo.chips:
        image = a.read_arena(coord)
        for name, prog in (("pre", a.pre), ("query", a.query), ("post", a.post)):
            words = a.proj_status(image, prog)
            if any(words[1:6]):
                out[("projections", name, coord)] = [hex(w) for w in words[:6]]
        _, _, _, flags = a.sm.pages_to_partials(ttnn.to_torch(out_pages[st.topo.chip(coord)]))
        if not bool((flags[:, :, 0] == STATUS_MAGIC).all()) or not bool((flags[:, :, 1] == 0).all()):
            out[("sparse_mla", coord)] = flags[:, :, :2].flatten().tolist()[:16]
    return out


def single_replay_ms(chain, trace_id, repeats=5):
    """ms of ONE replay that starts from the embedding streams (host write, synchronize, replay, synchronize), with
    the cost of a synchronize alone taken off."""
    mesh = chain.mesh
    idle = tracing.idle_sync_ms(mesh)
    times = []
    for _ in range(repeats):
        chain.reset_streams()
        start = tracing.sync(mesh)
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        times.append(1e3 * (tracing.sync(mesh) - start) - idle)
    return times, idle


@galaxy
def test_trace(mesh_device):
    """The step of 6 layers (19 programs) as ONE trace; then its latency and that of its parts."""
    chain = Chain(
        mesh_device,
        wait_limit=int(os.environ.get("HY4_CHAIN_WAIT", "30000000")),
        canary=os.environ.get("HY4_CHAIN_CANARY", "1") == "1",
    )
    st, mesh = chain.st, mesh_device
    try:
        chain.set_batch(0)
        programs = st.step_programs(audit=True)
        step = lambda: Stack.launch(programs)
        step()
        if is_mock():
            return
        ttnn.synchronize_device(mesh)
        status = full_status(chain)
        assert not status, f"a status is left after the untraced step: {status}"
        untraced = st.read_streams()
        head, lm = head_reference()
        label = f"chain P{PROMPT} [{MOE_FIDELITY}]"
        gold = chain.gold(chain.n - 1, "streams_out")
        emb = chain.gold(0, "streams_in")
        logger.info(
            f"HY4_MK11 {label} untraced step: streams vs HF pcc={pcc(gold, untraced):.6f}; canary {st.canary_summary()}"
        )

        chain.reset_streams()
        trace_id, _ = tracing.capture(mesh, step)
        ttnn.synchronize_device(mesh)
        try:
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
            logger.info(
                f"HY4_MK11 {label} trace of {len(programs)} programs (canary {'on' if st.canary is not None else 'off'}): trace buffer "
                f"{view.total_bytes_allocated_per_bank} B allocated per bank x {view.num_banks} banks per chip (region {view.total_bytes_per_bank} B per bank)"
            )
        except Exception as error:  # the view of the trace region is a convenience, not part of the test
            logger.info(f"HY4_MK11 {label} trace buffer size not readable: {str(error)[:200]}")
        status = st.status()
        assert not status, f"words not at rest after the capture: {status}"
        try:
            chain.reset_streams()
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            status = full_status(chain)
            assert not status, f"a status is left after the traced step: {status}"
            traced = st.read_streams()
            upd = lambda s: s - emb
            logger.info(
                f"HY4_MK11 {label} ONE trace of {len(programs)} programs, one replay from the embedding: streams after layer {chain.n - 1} vs the "
                f"untraced step's pcc={pcc(untraced, traced):.7f} max_abs={(untraced - traced).abs().max():.3e} (bit-equal: "
                f"{torch.equal(untraced, traced)}); vs HF pcc={pcc(gold, traced):.6f}, update pcc={pcc(upd(gold), upd(traced)):.5f} gain="
                f"{gain(upd(gold), upd(traced)):.4f}; program-cache entries {mesh.num_program_cache_entries()}"
            )
            res = check_logits(chain, head, lm, traced, f"{label} traced")
            canary = st.canary_summary()
            logger.info(
                f"HY4_MK11 {label} canary on the 7 heads after the untraced step and one replay (a capture runs nothing): {canary}; per head {st.read_canary()}"
            )
            for kind, v in canary.items():
                assert v["found"] == 0 and v["timeouts"] == 0 and v["checks"] == v["runs"], f"canary {kind}: {v}"
            # not bit for bit: X4 merges in arrival order, so two runs of the same step differ in ctx's last bits and
            # a near-tie expert pick can flip after it (PLAN.md section 15: 0.9995 to 0.9999 measured between replays)
            # The difference between two runs grows with depth (measured: 0.9995 to 0.9999 at 6 layers, 0.9987 at 24),
            # so the bound does too, and the two runs must stand equally against HF.
            assert pcc(untraced, traced) > 1 - 2.5e-4 * chain.n, "the traced step differs from the untraced one"
            assert bool(torch.isfinite(traced).all()), "the traced step's streams are not finite"
            assert abs(pcc(gold, traced) - pcc(gold, untraced)) < 2e-3 * max(
                1, chain.n / 12
            ), "the traced step stands differently against HF"
            if JOB == "A":
                logger.info(f"HY4_MK11 {label} job A done: capture and ONE replay, every status empty, canary clean")
                return

            # run to run: the same trace twice more from the same embedding
            again = []
            for _ in range(2):
                chain.reset_streams()
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                ttnn.synchronize_device(mesh)
                again.append(st.read_streams())
            rows = lambda x, y: [round(pcc(x[r], y[r]), 6) for r in range(8)]
            logger.info(
                f"HY4_MK11 {label} run to run, the same trace from the same embedding: replay 2 vs replay 1 pcc={pcc(traced, again[0]):.7f} "
                f"(bit-equal {torch.equal(traced, again[0])}, per row {rows(traced, again[0])}); replay 3 vs replay 2 pcc="
                f"{pcc(again[0], again[1]):.7f} (bit-equal {torch.equal(again[0], again[1])}); the untraced step vs replay 1 per row "
                f"{rows(untraced, traced)}; vs HF: {[round(pcc(gold, v), 6) for v in (untraced, traced, again[0], again[1])]}"
            )
            # latency of the whole step
            singles, idle = single_replay_ms(chain, trace_id)
            rounds = [tracing.replay_ms(mesh, trace_id, 20) for _ in range(3)]
            ttnn.synchronize_device(mesh)
            status = full_status(chain)
            after = st.read_streams()
            best = min(rounds)
            logger.info(
                f"HY4_MK11 {label} LATENCY step of {chain.n} layers, 8 rows, one trace: 20 replays back to back and one synchronize: "
                f"{best:.3f} ms per step (rounds {[round(v, 3) for v in rounds]}) = {best / chain.n * 1e3:.1f} us per layer = {8e3 / best:.0f} "
                f"tokens/s for {chain.n} layers; a single replay from the embedding {min(singles):.3f} ms (five: {[round(v, 3) for v in singles]}; "
                f"a synchronize alone {idle:.3f} ms, taken off); 78 layers at this rate {best * 78 / chain.n:.1f} ms = {8e3 / (best * 78 / chain.n):.1f} tokens/s "
                f"for 8 rows (shared-indexer layers only); words after the replays: {status or 'at rest'}; streams finite after 65 steps without a reset: "
                f"{bool(torch.isfinite(after).all())} (rms {after.pow(2).mean().sqrt():.3e})"
            )
            assert not status, f"a status is left after the replays: {status}"
            canary = st.canary_summary()
            logger.info(
                f"HY4_MK11 {label} canary after {canary.get('p2', {}).get('runs', 0) // 7 if canary else 0} P2 / MLP runs per head: {canary}"
            )
            for kind, v in canary.items():
                assert v["found"] == 0 and v["timeouts"] == 0 and v["checks"] == v["runs"], f"canary {kind}: {v}"
        finally:
            ttnn.release_trace(mesh, trace_id)

        # the parts: the attention programs alone, the MLP programs alone, prefixes of k layers
        parts = {
            "attention programs (P1, P2)": [
                p for p in programs if p[0].startswith("p") and not p[0].startswith("post")
            ],
            "MLP programs and POST": [p for p in programs if p[0].startswith("mlp") or p[0] == "post"],
        }
        every = 1 if chain.n <= 12 else 6
        for k in sorted(set(range(every, chain.n + 1, every)) | {1, chain.n}):
            parts[f"layers 0..{k - 1}"] = programs[: 3 * k]
        prev, prev_k, out = 0.0, 0, {}
        for name, subset in parts.items():
            chain.reset_streams()
            r = tracing.traced_ms(mesh, lambda: Stack.launch(subset))
            ttnn.synchronize_device(mesh)
            status = st.status()
            out[name] = r["replay_ms"]
            extra = ""
            if name.startswith("layers"):
                k = len(subset) // 3
                extra = (
                    f"; the layers since the prefix before: {1e3 * (r['replay_ms'] - prev) / (k - prev_k):.1f} us each"
                )
                prev, prev_k = r["replay_ms"], k
            logger.info(
                f"HY4_MK11 {label} LATENCY {name}: {len(subset)} programs, {r['replay_ms'] * 1e3:.1f} us per replay (rounds "
                f"{[round(v * 1e3, 1) for v in r['rounds_ms']]}){extra}; words {status or 'at rest'}"
            )
            assert not status, f"words not at rest after {name}: {status}"
        a_ms, m_ms, n = out["attention programs (P1, P2)"], out["MLP programs and POST"], chain.n
        logger.info(
            f"HY4_MK11 {label} LATENCY halves: attention {a_ms / n * 1e3:.1f} us per layer, MLP {m_ms / n * 1e3:.1f} us per layer (with "
            f"POST), sum {1e3 * (a_ms + m_ms) / n:.1f} us per layer against {1e3 * out[f'layers 0..{n - 1}'] / n:.1f} in the chain"
        )
    finally:
        chain.free()


def test_control():
    """Host only (run it with HY4_MOCK=1: no board). The reference with the checkpoint's exact weights, free-running from
    HF's embedding over the SAME cache rows the chips held (read back from the packed storage: the FP8 latents) and,
    as a check of the control itself, over HF's own rows; against HF and against the device's stages of test_accuracy
    (the dump). Columns per layer: HF vs the reference on HF's rows (the control's own error), HF vs the reference on the
    stored rows (FP8 rows alone), the reference on the stored rows vs the device (everything else of ours)."""
    from models.experimental.hy4_preview.fused.ihc_site.op import golden_post, golden_pre
    from models.experimental.hy4_preview.fused.layer_mlp.tests.test_mlp_layer import DenseWeights, Weights
    from models.experimental.hy4_preview.fused.sparse_mla.tests.test_attend import _checkpoint_reference
    from models.experimental.hy4_preview.reference.attention import rms_norm

    if not os.path.isfile(DUMP):
        pytest.skip(f"{DUMP}: run test_accuracy on the board first")
    dump = torch.load(DUMP, weights_only=False)
    g = hf_harness.find_decode_golden(LAYERS, PROMPT)
    N = g["meta"]["prompt_len"]
    cfg = IHCConfig()
    steps = dump["steps"]
    rows = [(b, u) for b in range(len(steps)) for u in range(8)]
    J = [steps[b]["J"][u] for b, u in rows]
    gold = lambda layer, stage: g["layers"][layer][stage][J].float()
    dev = lambda layer, stage: torch.cat([steps[b]["out"][(layer, stage)] for b in range(len(steps))])
    emb = gold(0, "streams_in")
    variants = {"hf_rows": emb.clone(), "stored_rows": emb.clone()}
    # HY4_CONTROL_SCALED=1: two more variants, the stored rows with their FP8 latent part (not the 64 RoPE values)
    # multiplied by one scalar at read: 1.0430 for every layer, and per layer 1 / (the measured gain of that layer's
    # stored latent part against HF's): what a factor folded into the weights would do to the truncating cast's bias
    scaled = os.environ.get("HY4_CONTROL_SCALED") == "1"
    if scaled:
        variants = {"stored_x1.0430": emb.clone(), "stored_x_per_layer": emb.clone()}
    # HY4_CONTROL_BFP4=1: the experts rounded to BFP4 as the decode streams hold them (routed experts one BFP4 term,
    # the shared expert and layer 0's dense FFN two; Weights.moe(rounded=True), the rounding ttnn's own host
    # conversion), everything else exact: on HF's own rows, and on the stored rows with the 1.0430 scalar
    bfp4 = os.environ.get("HY4_CONTROL_BFP4") == "1"
    if bfp4:
        variants = {"hf_rows_bfp4": emb.clone(), "stored_x1.0430_bfp4": emb.clone()}
        scaled = False
    tag = "bfp4 experts" if bfp4 else "scaled"
    LAT = 512
    result = {}
    kind = "bfp4" if bfp4 else ("scaled" if scaled else "exact")
    keep = {"J": J, "rows": rows, "layers": {}, "logits": {}}  # per row, for reading single rows afterwards
    for layer in range(LAYERS):
        ref = _checkpoint_reference(layer)
        (gamma_attn,) = hy4_tensors([f"model.layers.{layer}.input_layernorm.weight"])
        s_attn, s_mlp = site(layer, "hc_attn_layer"), site(layer, "hc_mlp_layer")
        w = DenseWeights(0) if layer == 0 else Weights(layer)
        stage = {}
        c_layer = 1.0
        if scaled:
            ref_lat = torch.cat(
                [g["cache"][layer]["latent"][: steps[b]["positions"][u] + 1, :LAT].float() for b, u in rows]
            )
            got_lat = torch.cat([steps[b]["stored"][(layer, u)][:, :LAT] for b, u in rows])
            c_layer = 1.0 / gain(ref_lat, got_lat)
        for name, streams in variants.items():
            y, _, post, _ = golden_pre(cfg, *s_attn, streams)
            x = rms_norm(y, gamma_attn.float(), cfg.norm_eps)
            outs = []
            with torch.no_grad():
                for i, (b, u) in enumerate(rows):
                    p = steps[b]["positions"][u]
                    lat = (
                        steps[b]["stored"][(layer, u)]
                        if name.startswith("stored")
                        else g["cache"][layer]["latent"][: p + 1].float()
                    )
                    if name.startswith("stored_x"):
                        lat = lat.clone()
                        lat[:, :LAT] *= 1.0430 if name.startswith("stored_x1.0430") else c_layer
                    token = ref.new_token(x[i : i + 1], p)
                    outs.append(ref.absorbed_stages(x[i : i + 1], token, lat)["out"].float())
            f_attn = torch.cat(outs)
            mid = golden_post(cfg, streams, f_attn, post)
            y2, _, post2, _ = golden_pre(cfg, *s_mlp, mid)
            x2 = rms_norm(y2, w.gamma.float(), cfg.norm_eps)
            rounded = name.endswith("_bfp4")
            if layer == 0:
                f_mlp, _ = w.ffn(x2, rounded=rounded)
                ids = None
            else:
                _, ids, weights = ML.route_golden(x2, w.router_w, w.router_b)
                if rounded:  # the picked experts and the shared one, rounded once per layer, 16 at a time
                    from concurrent.futures import ThreadPoolExecutor

                    todo = [
                        e for e in sorted(set(ids.flatten().tolist())) + [256] if e not in w._rounded
                    ]  # (256: the shared)
                    with ThreadPoolExecutor(16) as pool:
                        list(pool.map(w.rounded, todo))
                f_mlp = w.moe(x2, ids, weights, rounded=rounded)
            variants[name] = golden_post(cfg, mid, f_mlp, post2)
            stage[name] = {
                "attn_in": x,
                "attn_out": f_attn,
                "streams_mid": mid,
                "mlp_in": x2,
                "mlp_out": f_mlp,
                "streams_out": variants[name],
                "ids": ids,
            }
        del w
        keep["layers"][layer] = {
            name: {k: st_[k] for k in ("attn_out", "mlp_in", "mlp_out", "streams_out", "ids")}
            for name, st_ in stage.items()
        }
        pg = lambda a, b: f"{pcc(a, b):.5f}/{gain(a, b):.4f}"
        upd = lambda s: s - emb
        if scaled or bfp4:
            hf_ids = g["layers"][layer]["router_indices"][J] if layer else None
            same = lambda a, b: sum(set(a[r].tolist()) == set(b[r].tolist()) for r in range(len(rows)))
            text = "; ".join(
                f"{name}: attn_out {pg(gold(layer, 'attn_out'), st_['attn_out'])}, mlp_out {pg(gold(layer, 'mlp_out'), st_['mlp_out'])}, "
                f"streams_out {pg(gold(layer, 'streams_out'), st_['streams_out'])}"
                + (f", rows with HF's 8 experts {same(hf_ids, st_['ids'])}/{len(rows)}" if layer else "")
                for name, st_ in stage.items()
            )
            logger.info(
                f"HY4_MK11 control {tag} P{PROMPT} layer {layer} (vs HF, pcc/gain over {len(rows)} rows; per-layer factor {c_layer:.4f}): {text}"
            )
            continue
        line = []
        for what in ("attn_out", "mlp_out", "streams_out"):
            line.append(
                f"{what}: HF vs ref on HF's rows {pg(gold(layer, what), stage['hf_rows'][what])}, HF vs ref on the stored rows "
                f"{pg(gold(layer, what), stage['stored_rows'][what])}, ref on the stored rows vs device {pg(stage['stored_rows'][what], dev(layer, what))}, "
                f"HF vs device {pg(gold(layer, what), dev(layer, what))}"
            )
        so, sd, sg = stage["stored_rows"]["streams_out"], dev(layer, "streams_out"), gold(layer, "streams_out")
        line.append(
            f"update since the embedding: HF vs ref on the stored rows {pg(upd(sg), upd(so))}, ref on the stored rows vs device "
            f"{pg(upd(so), upd(sd))}, HF vs device {pg(upd(sg), upd(sd))}"
        )
        if layer:
            hf_ids = g["layers"][layer]["router_indices"][J]
            d_ids = torch.cat([steps[b]["out"][(layer, "picks")]["ids"].long() for b in range(len(steps))])
            same = lambda a, b: sum(set(a[r].tolist()) == set(b[r].tolist()) for r in range(len(rows)))
            line.append(
                f"picks, rows with the same 8 experts of {len(rows)}: HF vs ref on HF's rows {same(hf_ids, stage['hf_rows']['ids'])}, HF vs ref "
                f"on the stored rows {same(hf_ids, stage['stored_rows']['ids'])}, ref on the stored rows vs device {same(stage['stored_rows']['ids'], d_ids)}, "
                f"HF vs device {same(hf_ids, d_ids)}"
            )
        logger.info(
            f"HY4_MK11 control P{PROMPT} [{dump['fidelity']}] layer {layer} (pcc/gain over {len(rows)} rows): "
            + "; ".join(line)
        )
        result[layer] = stage
    head, lm = head_reference()
    gl = g["logits"][J].float()
    want = g["cached_logits_argmax"][J]
    for name, streams in list(variants.items()) + [("device", dev(LAYERS - 1, "streams_out"))]:
        _, logits = host_logits(head, lm, streams)
        top5 = sum(int(want[r]) in logits[r].topk(5).indices.tolist() for r in range(len(rows)))
        logger.info(
            f"HY4_MK11 control P{PROMPT} head on {name}: logits vs HF pcc={pcc(gl, logits):.6f} (per row min "
            f"{min(pcc(gl[r], logits[r]) for r in range(len(rows))):.6f}); top-1 equal to HF's on {int((logits.argmax(-1) == want).sum())} of "
            f"{len(rows)} rows, HF's token in the top 5 on {top5}"
        )
        got = logits.argmax(-1)
        margin = lambda r: float(gl[r].topk(2).values[0] - gl[r].topk(2).values[1])
        logger.info(
            f"HY4_MK11 control P{PROMPT} head on {name}, per row: differing (step, HF, got, HF's margin, row logits pcc) "
            f"{[(int(J[r]), int(want[r]), int(got[r]), round(margin(r), 3), round(pcc(gl[r], logits[r]), 4)) for r in range(len(rows)) if got[r] != want[r]]}; "
            f"row logits pcc {[round(pcc(gl[r], logits[r]), 4) for r in range(len(rows))]}"
        )
        keep["logits"][name] = logits
    try:
        path = DUMP[:-3] + f"_control_{kind}.pt"
        torch.save(keep, path)
        logger.info(f"HY4_MK11 control P{PROMPT}: per-row stages of {list(variants)} and the logits saved to {path}")
    except Exception as error:  # the lines above are the result; the file is for later reading
        logger.warning(f"HY4_MK11 control: per-row stages not saved: {error}")


@galaxy
def test_program_times(mesh_device):
    """Every program's time inside the six-layer trace, from prefixes of the trace (P1, P2 and MLP of each layer
    separately): prefix k = the first k programs as a trace of their own, 20 replays back to back; a program's time is
    the difference of two prefixes. A prefix that ends in a P1 leaves that P1's X3 arrivals counted on the landing
    cores (its P2 would take them): they are checked and zeroed after that prefix. No stamped build."""
    chain = Chain(mesh_device, wait_limit=30_000_000)
    st, a, mesh = chain.st, chain.st.a, mesh_device
    try:
        chain.set_batch(0)
        programs = st.step_programs(audit=True)
        Stack.launch(programs)
        if is_mock():
            return
        ttnn.synchronize_device(mesh)
        status = st.status()
        assert not status, f"words not at rest after the untraced step: {status}"
        label = f"chain P{PROMPT} [{MOE_FIDELITY}]"
        prev, rows = 0.0, {}
        for k in range(1, len(programs) + 1):
            subset = programs[:k]
            name = subset[-1][0]
            chain.reset_streams()
            r = tracing.traced_ms(mesh, lambda: Stack.launch(subset))
            ttnn.synchronize_device(mesh)
            if name.startswith("p1"):
                left = a.status(p1_only=True)
                # the landing cores hold one count per replay; anything else must be at rest
                other = {
                    key: v
                    for key, v in left.items()
                    if not (len(key) == 3 and key[1] in A.MERGE and set(v) == {A.W_QARR})
                }
                assert not other, f"words not at rest after the prefix ending in {name}: {other}"
                a.reset_words()
            status = st.status()
            assert not status, f"words not at rest after the prefix ending in {name}: {status}"
            rows[name] = 1e3 * (r["replay_ms"] - prev)
            logger.info(
                f"HY4_MK11 {label} PROGRAM {name}: {rows[name]:.1f} us (prefix of {k} programs {r['replay_ms'] * 1e3:.1f} us, rounds "
                f"{[round(v * 1e3, 1) for v in r['rounds_ms']]})"
            )
            prev = r["replay_ms"]
        for kind in ("p1", "p2", "mlp"):
            v = [rows[f"{kind}[{n}]"] for n in range(chain.n)]
            logger.info(
                f"HY4_MK11 {label} PROGRAM TABLE {kind} of layers 0..5: {[round(x, 1) for x in v]} us, mean {sum(v) / len(v):.1f}"
            )
        logger.info(f"HY4_MK11 {label} PROGRAM TABLE post: {rows['post']:.1f} us; canary {st.canary_summary()}")
    finally:
        chain.free()
