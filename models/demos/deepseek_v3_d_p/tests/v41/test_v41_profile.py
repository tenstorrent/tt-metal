# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-profiler and traced-timing runs of the production V4.1 blocks per prefill scenario (G2 per-op table, beads
8y7.9.3, 8y7.9.13).

Setup: the real-weight blocks of the sharing schedule 0 -> 2 -> 3 -> 20 -> 21 -> 24 (real bfp8 experts, LoudBox 2x4,
BF16 KV) run one chunk of real text at the scenario's chunk start. The first block's input is what
``TtV41Transformer`` builds on device (checkpoint embedding of the chunk's tokens, expanded to the hc streams; one-hot
pre-mix); every later layer takes the device outputs. Text: the held-out novel (``oracle.text_tokens``), repeated from
its start past its ``BOOK_TOKENS`` tokens (S2's chunk at 512000 is novel tokens 152000-157120).

Scenarios (``SCENARIOS``, the DeepSeek-V3.2 / GLM ``tests/sparse_mla/test_sparse_mla_perf.py`` set): a 5120-token
chunk at start 0 (empty cache), 51200 and 512000; ``slice_*``: V3.2's per-chip Galaxy slice on LoudBox (chunk and
start x sp / 8: chunk 1280 at 0 / 12800 / 128000). Cache fill before the measured chunk:

* ``forward``: real forwards of every earlier chunk of the text through the same six blocks (``state.advance``);
* ``tiled``: real forwards of the first ``start / TILE_PERIODS`` tokens, then those compressed-KV and index-K rows
  are copied on device ``TILE_PERIODS`` times to fill the cache up to ``start`` (the window carries are the last
  forward chunk's). Real-text value statistics at a tenth of the fill time; positions (RoPE) inside the copies
  repeat, which timing does not depend on. ``S1_tiled`` against ``S1`` validates it.

Tests:

* ``test_v41_block_profile``: only under the Tracy device profiler (``scripts/run_safe_pytest.sh --profile <id>``);
  phases delimited by Tracy signposts for ``g2_profile_table.py``: ``warm`` (compile), ``untraced`` (one forward per
  layer, every sublayer / MoE / attention stage and mHC call in ``B <stage>`` / ``E <stage>`` signposts), ``traced``
  (one warm + one measured trace replay per layer). Fill forwards carry no signposts (the attribution ignores them).
  The device profiler buffer is read after every forward / replay (``ttnn.ReadDeviceProfiler``); set
  ``TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT`` above one forward's program count.
* ``test_v41_scenario_traced``: unprofiled; per layer the traced replay wall (``TRACED_REPLAYS`` blocking replays:
  min and median; device time, the forward crosses no host boundary) and the model's optimistic / conservative /
  target (``utils/v41_perf_model``). ``V41_SCENARIO_TRACED`` JSON lines, also appended to ``$V41_SCENARIO_OUT``
  (default ``generated/v41_scenario_traced.log``). No pass/fail bar.
"""

import json
import os
import statistics
import time
from contextlib import contextmanager

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41 import expert_dtype_reference as R
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_expert_dtype import EXPERT_DTYPES, _weights
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_perf import LAYERS
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_trace import MESH, capture
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import mhc_expand
from models.demos.deepseek_v3_d_p.tt.v41.block import TtV41Block
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.head import TtV41Embedding
from models.demos.deepseek_v3_d_p.tt.v41.mhc import initial_pre_mix
from models.demos.deepseek_v3_d_p.tt.v41.weights import resolve_checkpoint
from models.demos.deepseek_v3_d_p.utils import v41_perf_model as M
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

EXPERT_DTYPE = "bfp8"
BOOK_TOKENS = 180_000  # the novel has 189,261 V4.1 tokens from its first chapter; wrap before its end
TILE_PERIODS = 10
TRACED_REPLAYS = 5
SCENARIOS = {  # id: (chunk, chunk start, cache fill)
    "S0": (5120, 0, "none"),
    "S1": (5120, 51200, "forward"),
    "S1_tiled": (5120, 51200, "tiled"),
    "S2": (5120, 512000, "tiled"),
    "S2_forward": (5120, 512000, "forward"),
    "slice_S0": (1280, 0, "none"),
    "slice_S1": (1280, 12800, "forward"),
    "slice_S2": (1280, 128000, "tiled"),
}
OUT = os.environ.get("V41_SCENARIO_OUT", "generated/v41_scenario_traced.log")


def _event(msg: str) -> None:
    """Stage log line (TT_EVENT) with a UTC timestamp, also appended to ``OUT`` so it survives the runner log."""
    logger.info(f"TT_EVENT {msg}")
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def scenario_tokens(total: int):
    """[total] int64 real-text tokens: the novel's first ``BOOK_TOKENS``, repeated."""
    book = orc.text_tokens(BOOK_TOKENS)[0]
    return book.repeat(-(-total // BOOK_TOKENS))[:total]


class _Signposted:
    """Wraps a callable in ``B <name>`` / ``E <name>`` signposts while ``flag['on']``."""

    def __init__(self, fn, name, flag):
        self._fn, self._name, self._flag = fn, name, flag

    def __call__(self, *args, **kwargs):
        if not self._flag["on"]:
            return self._fn(*args, **kwargs)
        signpost(f"B {self._name}")
        out = self._fn(*args, **kwargs)
        signpost(f"E {self._name}")
        return out

    def __getattr__(self, name):
        return getattr(self._fn, name)


def _instrument(block, flag):
    def wrap(owner, attr, name):
        if getattr(owner, attr, None) is not None:
            setattr(owner, attr, _Signposted(getattr(owner, attr), name, flag))

    attn, moe, res = block.attn, block.ffn.moe, block.residual
    for attr in ("compressor", "index_keys", "indexer"):
        wrap(attn, attr, f"attention.{attr}")
    for attr in ("gate", "dispatch_module", "routed_expert", "combine_module", "reduce_module", "shared_expert"):
        wrap(moe, attr, f"moe.{attr}")
    for site in ("attn_site", "ffn_site"):
        for attr in ("split", "collapse", "hc_post"):
            wrap(getattr(res, site), attr, f"mhc.{site}.{attr}")
    wrap(block, "attn_norm", "attn_norm")
    wrap(block, "attn", "attention")
    wrap(block, "ffn_norm", "ffn_norm")
    wrap(block, "ffn", "moe")


ZERO_UPLOAD_LIMIT = 64 << 20  # bytes of one host upload of a replicated zero cache tensor (per chip)


@contextmanager
def _device_zeroed_caches(mesh_device):
    """``V41PrefillState`` uploads its zero caches from the host; a replicated 0.5 GB KV tensor (S2's ratio-1 source)
    is a 4 GB host write that outlasts the safe runner's 5 s dispatch timeout (reported as a hang). While active, a
    zero upload above ``ZERO_UPLOAD_LIMIT`` is instead allocated on device and zeroed by device copies of one small
    uploaded zero block (same shape, dtype, layout and contents)."""
    upload = ttnn.from_torch

    def from_torch(t, *args, **kwargs):
        big = isinstance(t, torch.Tensor) and t.dim() == 4 and t.numel() * t.element_size() > ZERO_UPLOAD_LIMIT
        if not big or kwargs.get("device") is None or t.any():
            return upload(t, *args, **kwargs)
        rows, width = t.shape[2], t.shape[3]
        step = max(ZERO_UPLOAD_LIMIT // (width * t.element_size()), 32) // 32 * 32
        block = upload(torch.zeros(1, 1, step, width, dtype=t.dtype), *args, **kwargs)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, rows, width]), block.dtype, block.layout, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        for first in range(0, rows, step):
            n = min(step, rows - first)
            piece = block if n == step else ttnn.slice(block, [0, 0, 0, 0], [1, 1, n, width])
            ttnn.experimental.slice_write(piece, out, [0, 0, first, 0], [1, 1, first + n, width], [1, 1, 1, 1])
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(block)
        return out

    ttnn.from_torch = from_torch
    try:
        yield
    finally:
        ttnn.from_torch = upload


class ScenarioStack:
    """The six real-weight blocks for ``chunk``, the device embedding, the text and a state filled up to ``start``."""

    def __init__(self, mesh_device, scenario: str):
        self.mesh_device = mesh_device
        self.chunk, self.start, self.fill = SCENARIOS[scenario]
        ckpt = resolve_checkpoint()
        if ckpt is None:
            pytest.skip("V4.1 checkpoint shards not downloaded")
        shape = tuple(mesh_device.shape)
        spec = R.block_spec(LAYERS[0], LAYERS)  # the weight-cache directory is schedule independent
        t0 = time.perf_counter()
        self.blocks = {}
        for layer in LAYERS:
            start = time.perf_counter()
            root, w, marker = _weights(ckpt, layer, EXPERT_DTYPE, shape, spec)
            self.blocks[layer] = TtV41Block(
                mesh_device,
                C,
                layer,
                w,
                self.chunk,
                routed_expert_weights_dtype=EXPERT_DTYPES[EXPERT_DTYPE],
                weight_cache_path=root,
            )
            marker.touch()
            del w
            _event(f"block {layer} built {time.perf_counter() - start:.1f}s")
        self.embedding = TtV41Embedding(mesh_device, C, ckpt.read(["embed.weight"])["embed.weight"])
        self.tokens = scenario_tokens(self.start + self.chunk)
        self.pre0 = initial_pre_mix(mesh_device, C, self.chunk)
        with _device_zeroed_caches(mesh_device):
            self.state = V41PrefillState(
                mesh_device, C, self.start + self.chunk, self.chunk, list(LAYERS), kv_format=MlaKvCacheFormat.BF16_RM
            )
        _event(f"setup chunk {self.chunk} start {self.start} fill {self.fill}: {time.perf_counter() - t0:.1f}s")

    def chunk_input(self, first: int):
        """The first block's input (x, pre) for the chunk of tokens [first, first + chunk), built on device."""
        ids = ttnn.from_torch(
            self.tokens[first : first + self.chunk].to(torch.int32).view(1, 1, -1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, tuple(self.mesh_device.shape), dims=(2, None)),
        )
        x = mhc_expand(ttnn.typecast(self.embedding(ids), ttnn.float32), C.HC_MULT)
        ttnn.deallocate(ids)
        return x, self.pre0

    def forward(self, first: int, on_layer=None):
        """All six blocks on the chunk at ``first`` (= ``state.start``); ``on_layer(layer, x, pre)`` before each."""
        x, pre = self.chunk_input(first)
        for layer in LAYERS:
            if on_layer is not None:
                on_layer(layer, x, pre)
            x, pre = self.blocks[layer](x, pre, self.state, self.chunk)
        return x, pre

    def fill_cache(self):
        """Fill the state up to ``start`` (``fill``); afterwards ``state.start == start``."""
        if self.fill == "none":
            return
        end = self.start if self.fill == "forward" else self.start // TILE_PERIODS
        assert end % self.chunk == 0 and self.start % end == 0
        t0 = time.perf_counter()
        for first in range(0, end, self.chunk):
            t1 = time.perf_counter()
            self.forward(first)
            ttnn.synchronize_device(self.mesh_device)
            ttnn.ReadDeviceProfiler(self.mesh_device)
            self.state.advance(self.chunk)
            _event(f"fill chunk {first // self.chunk + 1}/{end // self.chunk} {time.perf_counter() - t1:.1f}s")
        if self.fill == "tiled":
            self._tile(end)
        assert self.state.start == self.start
        _event(f"fill {self.fill} to {self.start} done {time.perf_counter() - t0:.1f}s")

    def _tile(self, end: int):
        """Copy each KV source's compressed-KV and index-K rows of tokens [0, end) to fill tokens [end, start)."""
        s, g = self.state, self.state.geometry
        for source in s.kv:
            rows = end // C.compress_ratio(source)
            for cache, base in ((s.kv[source], g.kv_row_of_compressed(0)), (s.index_k[source], 0)):
                src = ttnn.slice(cache, [0, 0, base, 0], [1, 1, base + rows, cache.shape[3]])
                for k in range(1, self.start // end):
                    s._write_rows(cache, src, base + k * rows)
                ttnn.deallocate(src)
        ttnn.synchronize_device(self.mesh_device)
        s.advance(self.start - end)


SCENARIO_IDS = list(SCENARIOS)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("scenario", SCENARIO_IDS)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_block_profile(mesh_device, device_params, scenario):
    stack = ScenarioStack(mesh_device, scenario)
    blocks, chunk = stack.blocks, stack.chunk
    stack.fill_cache()
    flag = {"on": False}
    for layer in LAYERS:
        _instrument(blocks[layer], flag)

    def run(phase, inputs=None):
        x, pre = stack.chunk_input(stack.start)
        for layer in LAYERS:
            if inputs is not None:
                inputs[layer] = (x, pre)
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            signpost(f"B {phase} L{layer}")
            x, pre = blocks[layer](x, pre, stack.state, chunk)
            signpost(f"E {phase} L{layer}")
            ttnn.synchronize_device(mesh_device)
            ttnn.ReadDeviceProfiler(mesh_device)
            _event(f"{scenario} {phase} L{layer} {(time.perf_counter() - t0) * 1e3:.1f} ms (profiled)")

    t0 = time.perf_counter()
    run("warm")
    _event(f"{scenario} warm stack {time.perf_counter() - t0:.1f}s")
    flag["on"] = True
    run("untraced")
    flag["on"] = False

    inputs = {}
    run("staging", inputs=inputs)
    for layer in LAYERS:
        x, pre = inputs[layer]
        trace, _ = capture(mesh_device, lambda: blocks[layer](x, pre, stack.state, chunk), [blocks[layer].ffn])
        trace.replay()  # warm replay
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        t0 = time.perf_counter()
        signpost(f"B traced L{layer}")
        trace.replay()
        signpost(f"E traced L{layer}")
        ttnn.synchronize_device(mesh_device)
        _event(
            f"{scenario} traced L{layer} {(time.perf_counter() - t0) * 1e3:.1f} ms (profiled, {trace.num_segments} seg)"
        )
        ttnn.ReadDeviceProfiler(mesh_device)
        trace.release()


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("scenario", SCENARIO_IDS)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_scenario_traced(mesh_device, device_params, scenario):
    stack = ScenarioStack(mesh_device, scenario)
    blocks, chunk = stack.blocks, stack.chunk
    t_fill = time.perf_counter()
    stack.fill_cache()
    t_fill = time.perf_counter() - t_fill
    t0 = time.perf_counter()
    stack.forward(stack.start)  # compile at this start
    ttnn.synchronize_device(mesh_device)
    _event(f"{scenario} warm stack {time.perf_counter() - t0:.1f}s")
    inputs, untraced = {}, {}

    def keep(layer, x, pre):
        inputs[layer] = (x, pre)

    stack.forward(stack.start, on_layer=keep)  # inputs of every layer on this chunk
    ttnn.synchronize_device(mesh_device)
    for layer in LAYERS:
        x, pre = inputs[layer]
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        blocks[layer](x, pre, stack.state, chunk)
        ttnn.synchronize_device(mesh_device)
        untraced[layer] = (time.perf_counter() - t0) * 1e3
    workload = M.Workload(chunk=chunk, start=stack.start)
    layout = {(2, 4): M.LOUDBOX_2X4, (4, 2): M.LOUDBOX_4X2}[tuple(mesh_device.shape)]
    total = {"traced_min_ms": 0.0, "model_opt_ms": 0.0, "model_cons_ms": 0.0, "model_target_ms": 0.0}
    for layer in LAYERS:
        x, pre = inputs[layer]
        trace, _ = capture(mesh_device, lambda: blocks[layer](x, pre, stack.state, chunk), [blocks[layer].ffn])
        trace.replay()  # warm
        walls = []
        for _ in range(TRACED_REPLAYS):
            t0 = time.perf_counter()
            trace.replay()
            walls.append((time.perf_counter() - t0) * 1e3)
        trace.release()
        e = M.compose_block(layer, workload, layout)
        record = {
            "scenario": scenario,
            "fill": stack.fill,
            "chunk": chunk,
            "start": stack.start,
            "mesh": list(mesh_device.shape),
            "layer": layer,
            "type": C.block_type(layer).value,
            "traced_min_ms": round(min(walls), 3),
            "traced_median_ms": round(statistics.median(walls), 3),
            "traced_all_ms": [round(v, 3) for v in walls],
            "untraced_ms": round(untraced[layer], 3),
            "model_opt_ms": round(e.optimistic_ns / 1e6, 3),
            "model_cons_ms": round(e.conservative_ns / 1e6, 3),
            "model_target_ms": round(e.target_ns / 1e6, 3),
            "fill_s": round(t_fill, 1),
        }
        _event("V41_SCENARIO_TRACED " + json.dumps(record))
        for key in total:
            total[key] += record[key]
    _event(f"V41_SCENARIO_TRACED_TOTAL {scenario} " + json.dumps({k: round(v, 3) for k, v in total.items()}))
