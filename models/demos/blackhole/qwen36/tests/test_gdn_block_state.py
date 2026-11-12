# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: Qwen36ForCausalLM with the GDN request state in vLLM KV cache blocks (tt/gdn_block_state.py).

The adapter is driven as vllm-tt-plugin drives it (allocate_kv_cache_per_layer, prefill_forward and decode_forward
with mamba_state, zero_kv_cache_blocks), on a truncated model: 8 layers, attention at 3 and 7, so 2 layer slots and 3
GDN groups of 2 layers. The reference is the same model with its model-owned per-slot GDN state.

    MESH_DEVICE=P150x8 HF_MODEL=<Qwen3.8-27B dir> pytest models/demos/blackhole/qwen36/tests/test_gdn_block_state.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.tests.vllm_stubs import stub_vllm_if_missing
from models.demos.blackhole.qwen36.tests.gdn_block_helpers import PluginLayout
from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape, compute_pcc, model_path
from models.demos.blackhole.qwen36.tt.kv_cache_spec import HIST_SLOTS, GdnDims, pack_history_rows
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

N_LAYERS = 8
BMAX = 4
CTX = 4096
DECODE_STEPS = 4


def _parametrize(fn):
    shape = _resolve_mesh_shape(8)
    fn = pytest.mark.parametrize(
        "device_params",
        [
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "l1_small_size": GDN_CONV1D_L1_SMALL_SIZE,
                "trace_region_size": 268435456,
            }
        ],
        indirect=True,
    )(fn)
    return pytest.mark.parametrize("mesh_device", [pytest.param(shape, id=f"{shape[0]}x{shape[1]}")], indirect=True)(fn)


class Blocks:
    """Block ids of the requests: block 0 is the null block; request r owns ``bpu`` attention blocks and one state
    block per GDN group."""

    def __init__(self, layout, requests):
        self.layout = layout
        self.bpu = -(-CTX // layout.block_size)
        self.per_request = self.bpu + layout.num_groups
        self.num_blocks = 1 + requests * self.per_request

    def page_table(self, r):
        first = 1 + r * self.per_request
        return torch.arange(first, first + self.bpu, dtype=torch.int32)

    def state(self, r):
        first = 1 + r * self.per_request + self.bpu
        return tuple(range(first, first + self.layout.num_groups))


def _build(mesh_device):
    return Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=BMAX, max_seq_len=CTX, n_layers=N_LAYERS, hf_model=model_path(), enable_mtp=False
    )


def _adapter(monkeypatch, model, mesh_device, requests):
    stub_vllm_if_missing(monkeypatch)
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

    gen = Qwen36ForCausalLM([model], [model.args], mesh_device)
    probe = PluginLayout.for_model(model, num_blocks=1)
    blocks = Blocks(probe, requests)
    layout = PluginLayout.for_model(model, num_blocks=blocks.num_blocks)
    blocks.layout = layout
    kv = gen.allocate_kv_cache_per_layer(layout.per_layer_specs)
    return gen, layout, blocks, kv


def _prompts(n, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 20000, (n, T), generator=g, dtype=torch.int64)


def _decode_flags():
    return dict(
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )


def _device_tensors(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _pages(view_host):
    """[P, 32, 32] tiles of a float32 view [P / 8, 1, 64, 128]."""
    n = view_host.shape[0]
    return view_host.reshape(n, 2, 32, 4, 32).permute(0, 1, 3, 2, 4).reshape(-1, 32, 32)


def _head(pages, first):
    """[Dk, Dv] = [128, 128] from the 16 tiles starting at ``first`` (tile order as in [Dk, Dv])."""
    return pages[first : first + 16].reshape(4, 4, 32, 32).permute(0, 2, 1, 3).reshape(128, 128)


def read_block_state(model, layer, block):
    """Per device: (recurrent [Nv, Dk, Dv] float32, packed history [Nv, 4, 32, 32] bfloat16) of ``block``."""
    dn = model.layers[layer].attention
    sb, lay = dn.state_blocks, model.gdn_blocks.layout
    rec_k = [_pages(t) for t in _device_tensors(sb.rec_k)]
    rec_v = [_pages(t) for t in _device_tensors(sb.rec_v)] if sb.rec_v is not None else None
    hist = [t.reshape(-1, 32, 32) for t in _device_tensors(sb.hist)]
    out = []
    first = lay.first_fp32_page(block)
    for d in range(len(rec_k)):
        heads = []
        for h in range(lay.dims.nv):
            if h < lay.heads_k:
                heads.append(_head(rec_k[d], first + 16 * h))
            else:
                heads.append(_head(rec_v[d], first + 16 * (h - lay.heads_k)))
        hp = lay.hist_page(block, 0)
        out.append((torch.stack(heads), hist[d][hp : hp + lay.dims.nv * HIST_SLOTS].reshape(lay.dims.nv, 4, 32, 32)))
    return out


def _dims(model):
    a = model.args
    return GdnDims(nk=a.gdn_nk_tp, nv=a.gdn_nv_tp, dk=a.gdn_dk, dv=a.gdn_dv, kconv=a.gdn_conv_kernel_size)


def reference_state(model, layer, row):
    """Per device: (recurrent [Nv, Dk, Dv], packed history of the last 3 conv inputs) of model-owned slot ``row``."""
    dn = model.layers[layer].attention
    recs = _device_tensors(dn.rec_state)
    taps = [_device_tensors(t) for t in dn.conv_states]  # K x devices x [1, B, C]
    table = pack_history_rows(_dims(model))
    out = []
    for d, rec in enumerate(recs):
        carry = torch.stack([taps[j][d][0, row] for j in range(1, dn.K)])  # [K - 1, C], oldest first
        packed = carry.reshape(-1, 32)[table].reshape(-1, HIST_SLOTS, 32, 32)
        out.append((rec[row].float(), packed))
    return out


def _compare_state(got, want, rows_used, label, exact=False):
    for d, ((rec_g, hist_g), (rec_w, hist_w)) in enumerate(zip(got, want)):
        if exact:
            assert torch.equal(rec_g, rec_w), f"{label}: recurrent state differs on device {d}"
            assert torch.equal(hist_g[:, 1:, :rows_used], hist_w[:, 1:, :rows_used]), f"{label}: history, device {d}"
        else:
            pcc = compute_pcc(rec_g, rec_w)
            assert pcc >= 0.999, f"{label}: recurrent state PCC {pcc} on device {d}"
            pcc = compute_pcc(hist_g[:, 1:, :rows_used].float(), hist_w[:, 1:, :rows_used].float())
            assert pcc >= 0.999, f"{label}: history PCC {pcc} on device {d}"


def _gdn_layers(model):
    return [i for i, layer in enumerate(model.layers) if not layer.is_full_attention]


@torch.no_grad()
@_parametrize
@pytest.mark.parametrize("n_requests", [1, 3])
def test_block_state_matches_model_owned(mesh_device, n_requests, monkeypatch, reset_seeds, ensure_gc):
    """Prefill n requests and decode DECODE_STEPS steps (teacher-forced with the reference's greedy tokens): logits and
    every GDN layer's state in the blocks match the model-owned path. After prefill the state is a copy of the same
    computation, so it is bit-exact; decode runs gdn_decode_step (float32 gates and state) in place of the composite
    recurrence, so it is compared by PCC. Rows past n are padding rows on block 0."""
    model = _build(mesh_device)
    vocab = model.args.vocab_size
    lens = [300, 2048 + 77, 129][:n_requests]
    prompts = _prompts(n_requests, max(lens))
    layout0 = PluginLayout.for_model(model, num_blocks=1)
    blocks = Blocks(layout0, n_requests)
    pt = torch.zeros(BMAX, blocks.bpu, dtype=torch.int32)
    for r in range(n_requests):
        pt[r] = blocks.page_table(r)

    # ---- reference: model-owned per-slot state
    kv_shape = (blocks.num_blocks, model.args.n_local_kv_heads, layout0.block_size, model.args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    ref_prefill = model.prefill_paged_slots(
        [prompts[r : r + 1, : lens[r]] for r in range(n_requests)], pt[:n_requests], list(range(n_requests)), lens
    )
    ref_prefill = torch.cat([lg.reshape(1, -1)[:, :vocab] for lg in ref_prefill])
    tokens = [ref_prefill.argmax(-1)]
    gdn = _gdn_layers(model)
    ref_after_prefill = {(i, r): reference_state(model, i, r) for i in gdn for r in range(n_requests)}
    ref_decode = []
    pos = torch.full((BMAX,), -1, dtype=torch.int32)
    pos[:n_requests] = torch.tensor(lens, dtype=torch.int32)
    for step in range(DECODE_STEPS):
        tok = torch.zeros(BMAX, 1, dtype=torch.int32)
        tok[:n_requests, 0] = tokens[-1]
        dev = model.prepare_inputs_decode(tok, pos.clamp(min=0), pt)
        out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        lg = model.process_output_decode(out, BMAX)[:n_requests, 0, :vocab].float()
        ref_decode.append(lg)
        tokens.append(lg.argmax(-1))
        pos[:n_requests] += 1
    ref_after_decode = {(i, r): reference_state(model, i, r) for i in gdn for r in range(n_requests)}
    model.free_kv_caches()

    # ---- GDN state in vLLM blocks, through the adapter
    gen, layout, blocks2, kv = _adapter(monkeypatch, model, mesh_device, n_requests)
    assert blocks2.num_blocks == blocks.num_blocks
    states = [blocks.state(r) for r in range(n_requests)]
    group_of = {i: g for g, layers in enumerate(layout.gdn_groups) for i in layers}
    used = 2 * (2 * model.args.gdn_dk + model.args.gdn_dv) // 32
    logits, _ = gen.prefill_forward(
        prompts[:n_requests],
        pt[:n_requests],
        kv,
        lens,
        start_pos=[0] * n_requests,
        mamba_state=layout.step_state(states, BMAX),
    )
    assert torch.equal(logits[:, 0, :vocab], ref_prefill), "prefill logits differ from the model-owned path"
    for i in gdn:
        for r in range(n_requests):
            got = read_block_state(model, i, states[r][group_of[i]])
            _compare_state(got, ref_after_prefill[(i, r)], used, f"after prefill: layer {i} row {r}", exact=True)

    pos = torch.full((BMAX,), -1, dtype=torch.int32)
    pos[:n_requests] = torch.tensor(lens, dtype=torch.int32)
    rows = states + [None] * (BMAX - n_requests)
    for step in range(DECODE_STEPS):
        tok = torch.zeros(BMAX, 1, dtype=torch.int32)
        tok[:n_requests, 0] = tokens[step]
        out = gen.decode_forward(
            tok,
            pos,
            page_table=pt,
            kv_cache=kv,
            enable_trace=False,
            read_from_device=True,
            mamba_state=layout.step_state(rows, BMAX),
            **_decode_flags(),
        )
        lg = out[:n_requests, -1, :vocab].float()
        pcc = compute_pcc(lg, ref_decode[step])
        agree = (lg.argmax(-1) == ref_decode[step].argmax(-1)).float().mean().item()
        logger.info(f"decode step {step}: logits PCC {pcc:.6f}, greedy agreement {agree:.2f}")
        assert pcc >= 0.99, f"decode step {step}: logits PCC {pcc}"
        pos[:n_requests] += 1
    for i in gdn:
        for r in range(n_requests):
            got = read_block_state(model, i, states[r][group_of[i]])
            _compare_state(got, ref_after_decode[(i, r)], used, f"after decode: layer {i} row {r}")


@torch.no_grad()
@_parametrize
def test_chunked_prefill_block_reuse_and_zeroing(mesh_device, monkeypatch, reset_seeds, ensure_gc):
    """(1) A prompt prefilled in two vLLM chunks (the second at start_pos > 0 recomputes from position 0) leaves the
    same logits and block state as one prefill. (2) A request prefilled into blocks that a finished request left full
    of state gets the same logits and state as in fresh blocks: start_pos 0 never reads the block. (3)
    zero_kv_cache_blocks zeroes exactly the listed blocks."""
    model = _build(mesh_device)
    vocab = model.args.vocab_size
    gen, layout, blocks, kv = _adapter(monkeypatch, model, mesh_device, 3)
    used = 2 * (2 * model.args.gdn_dk + model.args.gdn_dv) // 32
    L, s = 2048 + 300, 2048
    prompt = _prompts(1, L, seed=1)

    def prefill(r, tokens, end, start=0):
        return gen.prefill_forward(
            tokens,
            blocks.page_table(r).reshape(1, -1),
            kv,
            [end],
            start_pos=[start],
            mamba_state=layout.step_state([blocks.state(r)], BMAX),
        )[0][0, 0, :vocab]

    whole = prefill(0, prompt, L)
    prefill(1, prompt[:, :s], s)
    chunked = prefill(1, prompt, L, start=s)
    assert torch.equal(whole, chunked)
    for i in _gdn_layers(model):
        g = next(g for g, layers in enumerate(layout.gdn_groups) if i in layers)
        _compare_state(
            read_block_state(model, i, blocks.state(1)[g]),
            read_block_state(model, i, blocks.state(0)[g]),
            used,
            "",
            True,
        )

    # Request 1 finished; request B reuses its blocks (still holding request 1's state). Request 2's blocks are fresh.
    other = _prompts(1, 700, seed=2)
    reused = prefill(1, other, 700)
    fresh = prefill(2, other, 700)
    assert torch.equal(reused, fresh)
    for i in _gdn_layers(model):
        g = next(g for g, layers in enumerate(layout.gdn_groups) if i in layers)
        _compare_state(
            read_block_state(model, i, blocks.state(1)[g]),
            read_block_state(model, i, blocks.state(2)[g]),
            used,
            "",
            True,
        )

    # Zero one state block and one attention block of request 0; every other block keeps its bytes.
    k0 = kv[0][0]
    before = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(k0)]
    victims = [blocks.state(0)[0], int(blocks.page_table(0)[1])]
    gen.zero_kv_cache_blocks(kv_cache=kv, block_ids=victims)
    after = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(k0)]
    for b_dev, a_dev in zip(before, after):
        for blk in range(blocks.num_blocks):
            if blk in victims:
                assert torch.count_nonzero(a_dev[blk]) == 0, f"block {blk} not zeroed"
            else:
                assert torch.equal(a_dev[blk], b_dev[blk]), f"block {blk} changed"


@torch.no_grad()
@_parametrize
def test_row_moves_and_traced_decode(mesh_device, monkeypatch, reset_seeds, ensure_gc):
    """After the plugin's two-phase warmup, traced decode follows the page index: three requests decode with their rows
    permuted at every step (slot_remap and mamba_state move together) and give the same logits per request as an
    unpermuted eager run."""
    model = _build(mesh_device)
    vocab = model.args.vocab_size
    gen, layout, blocks, kv = _adapter(monkeypatch, model, mesh_device, 3)
    n = 3
    lens = [200, 513, 1100]
    prompts = _prompts(n, max(lens), seed=3)
    pts = torch.stack([blocks.page_table(r) for r in range(n)])

    def run(perms, enable_trace):
        logits, _ = gen.prefill_forward(
            prompts,
            pts,
            kv,
            lens,
            start_pos=[0] * n,
            mamba_state=layout.step_state([blocks.state(r) for r in range(n)], BMAX),
        )
        tok = logits[:, 0, :vocab].argmax(-1)
        pos = torch.tensor(lens, dtype=torch.int32)
        out_per_request = []
        prev = list(range(n)) + [None] * (BMAX - n)
        for perm in perms:
            rows = [perm[i] if i < n else None for i in range(BMAX)]  # request held by each row
            remap = None
            if rows != prev:
                remap = torch.tensor(
                    [prev.index(r) if r is not None else i for i, r in enumerate(rows)], dtype=torch.int32
                )
            t = torch.zeros(BMAX, 1, dtype=torch.int32)
            p = torch.full((BMAX,), -1, dtype=torch.int32)
            pt = torch.zeros(BMAX, blocks.bpu, dtype=torch.int32)
            for i, r in enumerate(rows):
                if r is not None:
                    t[i, 0], p[i], pt[i] = tok[r], pos[r], pts[r]
            out = gen.decode_forward(
                t,
                p,
                page_table=pt,
                kv_cache=kv,
                enable_trace=enable_trace,
                read_from_device=True,
                slot_remap=remap,
                mamba_state=layout.step_state([blocks.state(r) if r is not None else None for r in rows], BMAX),
                **_decode_flags(),
            )
            lg = torch.zeros(n, vocab)
            for i, r in enumerate(rows):
                if r is not None:
                    lg[r] = out[i, -1, :vocab].float()
            out_per_request.append(lg)
            tok = lg.argmax(-1)
            pos += 1
            prev = rows
        return out_per_request

    identity = [[0, 1, 2]] * 4
    eager = run(identity, enable_trace=False)
    prefill_kw = dict(kv_cache=kv, can_sample_on_device=False)
    decode_kw = dict(kv_cache=kv, max_batch_size=BMAX, num_blocks=blocks.bpu, can_sample_on_device=False)
    gen.warmup_model_prefill(enable_trace=False, **prefill_kw)
    gen.warmup_model_decode(enable_trace=False, **decode_kw)
    gen.warmup_model_prefill(enable_trace=True, **prefill_kw)
    gen.warmup_model_decode(enable_trace=True, **decode_kw)
    moved = run([[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 1, 2]], enable_trace=True)
    for step, (a, b) in enumerate(zip(eager, moved)):
        pcc = compute_pcc(a, b)
        logger.info(f"step {step}: traced + moved vs eager logits PCC {pcc:.6f}")
        assert pcc >= 0.999 and torch.equal(a.argmax(-1), b.argmax(-1)), f"step {step}: PCC {pcc}"
