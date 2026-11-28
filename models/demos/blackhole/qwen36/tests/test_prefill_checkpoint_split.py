# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the prefill checkpoint of TPGatedDeltaNet.forward_prefill computed by TPGatedDeltaNet._delta_rule_split (two
chunk_gated_delta_rule calls split at the checkpoint, and one identity update chunk on the state there) against the
single call with beta and g zero past the checkpoint, next to the main call, that it replaces.

Qwen36ForCausalLM.prefill_forward with the GDN state in vLLM KV cache blocks (mamba_cache_mode "align" block ids,
1024-token blocks) prefills rows of 2000 tokens (checkpoint after 1024 inside the forward), 2348 (checkpoint after 2048:
a two-part plan) and 3100 (checkpoint after 3072 inside the masked tail). Every byte the rows leave must match: the
logits, the attention K and V of their pages, and the GDN recurrent state and convolution history in their destination
and checkpoint blocks.

    MESH_DEVICE=P150x8 HF_MODEL=<Qwen3.8-27B dir> pytest models/demos/blackhole/qwen36/tests/test_prefill_checkpoint_split.py

QWEN36_CKPT_DUMP=<file> makes test_traced_rows_dump save what the traced rows leave, so that two builds can be compared
byte for byte:

    python -c "import sys, torch; from models.demos.blackhole.qwen36.tests.test_prefill_checkpoint_split import \
        compare_dumps; print(compare_dumps(torch.load(sys.argv[1]), torch.load(sys.argv[2])))" A.64 B.64
"""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.tests.vllm_stubs import stub_vllm_if_missing
from models.demos.blackhole.qwen36.tests.gdn_block_helpers import PluginLayout
from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape, model_path
from models.demos.blackhole.qwen36.tests.test_gdn_block_state import BMAX, CTX, _device_tensors, _prompts
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

LENS = (2000, 2348, 3100)


def _parametrize(fn):
    """The TP=8 mesh with the serving launch's 1 GiB trace region: the 64-layer model's prefill traces do not fit the
    256 MiB of the truncated-model tests."""
    shape = _resolve_mesh_shape(8)
    fn = pytest.mark.parametrize(
        "device_params",
        [
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "l1_small_size": GDN_CONV1D_L1_SMALL_SIZE,
                "trace_region_size": 1 << 30,
            }
        ],
        indirect=True,
    )(fn)
    return pytest.mark.parametrize("mesh_device", [pytest.param(shape, id=f"{shape[0]}x{shape[1]}")], indirect=True)(fn)


def _single_call_checkpoint(self, delta_fn, call, qkvbg, init_state, checkpoint, valid_len, masks):
    """The checkpoint as one chunk_gated_delta_rule call over all rows with beta and g zero from checkpoint.length on,
    then the main call: the computation that TPGatedDeltaNet._delta_rule_split replaces, with the same operations in
    the same order. Eager only (the gate mask is a host write)."""
    q, k, v, beta, g = qkvbg
    mask = torch.zeros(1, q.shape[1], 1, dtype=torch.float32)
    mask[:, : checkpoint.length] = 1.0
    gate = ttnn.from_torch(
        mask,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=self.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
    )
    ck_beta = ttnn.multiply(beta, gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ck_g = ttnn.multiply(g, gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ck_o, ck_rec = delta_fn(q, k, v, ck_beta, ck_g, initial_state=init_state, **call)
    for t in (ck_o, ck_beta, ck_g, gate):
        ttnn.deallocate(t)
    main = dict(call)
    if masks is not None:
        main["masks"] = (masks.gate_mask, masks.qkv_mask)
    o, final_state = delta_fn(q, k, v, beta, g, initial_state=init_state, valid_len=valid_len, **main)
    return o, final_state, ck_rec


class Rows:
    """Qwen36ForCausalLM on a model of ``n_layers`` layers with GDN block state for ``n`` requests in KV cache blocks of
    ``block_size`` tokens (the serving launch's 1024 by default): request r owns ``bpu`` attention pages, a destination
    and a checkpoint block per GDN group."""

    def __init__(self, mesh_device, monkeypatch, n_layers, n, bmax=BMAX, block_size=1024):
        stub_vllm_if_missing(monkeypatch)
        from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

        self.mesh, self.bmax = mesh_device, bmax
        self.model = Qwen36Model.from_pretrained(
            mesh_device,
            max_batch_size=bmax,
            max_seq_len=CTX,
            n_layers=n_layers,
            hf_model=model_path(),
            enable_mtp=False,
        )
        self.gen = Qwen36ForCausalLM([self.model], [self.model.args], mesh_device)
        probe = PluginLayout.for_model(self.model, num_blocks=1, block_size=block_size)
        self.bs, self.groups = probe.block_size, probe.num_groups
        self.bpu = -(-CTX // self.bs)
        per = self.bpu + 2 * self.groups
        self.layout = PluginLayout.for_model(self.model, num_blocks=1 + n * per, block_size=block_size)
        self.kv = self.gen.allocate_kv_cache_per_layer(self.layout.per_layer_specs)
        first = [1 + r * per for r in range(n)]
        self.pages = torch.stack([torch.arange(f, f + self.bpu, dtype=torch.int32) for f in first])
        self.dest = [tuple(f + self.bpu + g for g in range(self.groups)) for f in first]
        self.ck = [tuple(f + self.bpu + self.groups + g for g in range(self.groups)) for f in first]

    def checkpoint(self, length, r):
        pos = (length - 1) // self.bs * self.bs  # vLLM's align-mode checkpoint: the last block boundary before the end
        return (pos, self.ck[r]) if pos else None

    def warmup(self):
        for enable_trace in (False, True):
            self.gen.warmup_model_prefill(enable_trace=enable_trace, kv_cache=self.kv, can_sample_on_device=False)

    def prefill(self, lens, rows, prompts):
        """One prefill step of ``rows`` (request r prefills lens[i] tokens of prompts[r]); host logits [n, vocab]."""
        logits, _ = self.gen.prefill_forward(
            prompts[rows],
            self.pages[rows],
            self.kv,
            list(lens),
            start_pos=[0] * len(rows),
            mamba_state=self.layout.step_state(
                [self.dest[r] for r in rows],
                self.bmax,
                checkpoints=[self.checkpoint(n, r) for n, r in zip(lens, rows)],
            ),
        )
        return logits[:, 0, : self.model.args.vocab_size].float()

    def snapshot(self, lens, rows, logits):
        """{label: [per-device host tensors]} of everything the rows wrote."""
        out = {f"row {r} logits": [logits[i].clone()] for i, r in enumerate(rows)}
        for c, (k, v) in enumerate(self.kv):
            for kind, t in (("K", k), ("V", v)):
                per_dev = _device_tensors(t)
                for r in rows:
                    out[f"row {r} attention {c} {kind}"] = [x[self.pages[r].long()].clone() for x in per_dev]
                del per_dev
        from models.demos.blackhole.qwen36.tests.test_mtp_gdn_blocks import block_state

        for i, layer in enumerate(self.model.layers):
            if layer.is_full_attention:
                continue
            g = next(g for g, members in enumerate(self.layout.gdn_groups) if i in members)
            for n, r in zip(lens, rows):
                blocks = [("end state", self.dest[r][g])]
                if self.checkpoint(n, r):
                    blocks.append(("checkpoint", self.ck[r][g]))
                for label, block in blocks:
                    st = block_state(self.model, i, block)
                    out[f"row {r} layer {i} {label} recurrent"] = [a.clone() for a, _ in st]
                    out[f"row {r} layer {i} {label} history"] = [b.clone() for _, b in st]
        return out


def _bytes(t):
    return t.contiguous().view(torch.uint8)


def compare_dumps(got, want):
    """[(label, device, number of differing bytes, largest absolute difference)] of two snapshots."""
    diffs = []
    assert sorted(got) == sorted(want), "the snapshots hold different entries"
    for label in sorted(want):
        for d, (g, w) in enumerate(zip(got[label], want[label], strict=True)):
            if g.shape != w.shape or g.dtype != w.dtype:
                diffs.append((label, d, -1, float("inf")))
                continue
            gb, wb = _bytes(g), _bytes(w)
            if not torch.equal(gb, wb):
                n = int((gb != wb).sum())
                worst = float((g.float() - w.float()).abs().nan_to_num(float("inf")).max())
                diffs.append((label, d, n, worst))
    return diffs


def _report(diffs, what):
    for label, d, n, worst in diffs[:20]:
        logger.error(f"{what}: {label} device {d}: {n} bytes differ, largest difference {worst:.3e}")
    return len(diffs)


@torch.no_grad()
@pytest.mark.timeout(7200)
@_parametrize
@pytest.mark.parametrize("n_layers", [8, 64], ids=["layers8", "layers64"])
def test_split_checkpoint_matches_the_single_call(mesh_device, n_layers, monkeypatch, reset_seeds, ensure_gc):
    """Three rows (2000, 2348 and 3100 tokens) in one eager prefill step, twice into the same blocks: with the
    single-call checkpoint (the reference) and with the split delta rule. The split run must leave the reference's
    bytes. The captured-trace path is compared across builds with test_traced_rows_dump: its chunk trace runs other
    programs than the eager chunk path, so a traced run does not match an eager one byte for byte."""
    from models.demos.blackhole.qwen36.tt.gdn import tp as gdn_tp

    rows = list(range(len(LENS)))
    t = Rows(mesh_device, monkeypatch, n_layers, len(rows))
    prompts = _prompts(len(rows), max(LENS), seed=71)
    split = gdn_tp.TPGatedDeltaNet._delta_rule_split
    with monkeypatch.context() as m:
        m.setattr(gdn_tp.TPGatedDeltaNet, "_delta_rule_split", _single_call_checkpoint)
        want = t.snapshot(LENS, rows, t.prefill(LENS, rows, prompts))
    assert gdn_tp.TPGatedDeltaNet._delta_rule_split is split
    got = t.snapshot(LENS, rows, t.prefill(LENS, rows, prompts))
    bad = _report(compare_dumps(got, want), "split vs single call")
    logger.info(f"CHECKPOINT SPLIT ({n_layers} layers): {len(want)} entries; the split differs in {bad}")
    assert bad == 0


@torch.no_grad()
@pytest.mark.timeout(7200)
@_parametrize
@pytest.mark.parametrize("n_layers", [8, 64], ids=["layers8", "layers64"])
def test_traced_rows_dump(mesh_device, n_layers, monkeypatch, reset_seeds, ensure_gc):
    """The three rows from the captured traces after the plugin's two-phase warmup, saved to QWEN36_CKPT_DUMP
    (suffixed with the layer count) so that the snapshots of two builds can be compared with compare_dumps. Uses only
    Qwen36ForCausalLM.prefill_forward, so it runs on a build without _delta_rule_split too."""
    path = os.environ.get("QWEN36_CKPT_DUMP")
    if not path:
        pytest.skip("QWEN36_CKPT_DUMP names no file")
    rows = list(range(len(LENS)))
    t = Rows(mesh_device, monkeypatch, n_layers, len(rows))
    prompts = _prompts(len(rows), max(LENS), seed=71)
    t.warmup()
    torch.save(t.snapshot(LENS, rows, t.prefill(LENS, rows, prompts)), f"{path}.{n_layers}")


@torch.no_grad()
@pytest.mark.timeout(7200)
@_parametrize
def test_checkpoint_burst_timing(mesh_device, monkeypatch, reset_seeds, ensure_gc):
    """Wall time of one prefill step of 7 rows of 2000 tokens (checkpoint after 1024 inside each forward: the first
    wave's burst at c8) on the full 64-layer model from the captured traces, max_num_seqs 8, median of 5 steps after one
    warm step (CHECKPOINT BURST line). Uses only Qwen36ForCausalLM.prefill_forward, so it runs on a build without
    _delta_rule_split too. Not a pass / fail gate."""
    n = 7
    rows = list(range(n))
    t = Rows(mesh_device, monkeypatch, 64, n, bmax=8)
    prompts = _prompts(n, 2000, seed=72)
    t.warmup()
    times = []
    for step in range(6):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        t.prefill([2000] * n, rows, prompts)
        ttnn.synchronize_device(mesh_device)
        if step:
            times.append(time.perf_counter() - t0)
    med = statistics.median(times) * 1e3
    logger.info(
        f"CHECKPOINT BURST: {n} rows of 2000 tokens in one step: {med:.1f} ms median ({med / n:.1f} ms per row), "
        f"steps {[round(x * 1e3, 1) for x in times]}"
    )
