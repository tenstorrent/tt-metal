# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 single-token decode on the prefill state (harness; eager, or traced per position parity).

``TtV41Decoder`` appends one token to a ``V41PrefillState`` that ``TtV41Transformer.prefill`` filled and
returns the
next-token logits, following the reference's ``start_pos > 0`` path (``inference/model.py``): the window ring,
the
compressor pooling at ``(start_pos + 1) % ratio == 0`` from the carried fp32 projections, the compressed row
and index
key written at ``start_pos // ratio`` with the group-first RoPE, the indexer over the ``(start_pos + 1) //
ratio``
visible rows, one sparse attention over the window and the selected rows.

Layout. The token is replicated on every row of one 32-row tile and on every chip (SP ranks run the same tile;
TP
splits hidden and heads as in prefill), so every fused V4.1 op runs in whole tiles unchanged and no query
split or
head->sequence all-to-all is needed: ``q`` is all-gathered over TP, every chip attends with all heads, the
output is
partitioned back by group. The caches are already replicated on every chip in natural order (``cache.py``), so
one
token reads the whole context without a collective. The MoE runs on a second ``TtV41Moe`` built for
``MOE_ROWS`` rows
per chip, sharing the expert tensors of the prefill block: 64 rows (the tile repeated) with the prefill
routing setup,
which counts tokens in multiples of 64, or the tile's 32 with the decode routing of ``tt/v41/routing.py``
(``V41_DECODE_ROUTING=1``: the gate's selection one core per row, the routing setup from one all-gathered
index tile)
and the decode MoE tail of ``tt/v41/moe_tail.py`` (``V41_DECODE_TAIL=1``: the combine's weighted sum on 80
cores).

Fixed shapes. Every step runs the same device graph for its position parity, so it can be traced: the indexer
scores
the whole index-K cache (``-inf`` added to the columns not yet written, a mask row the host uploads per step)
and
selects ``INDEX_TOPK`` rows (sentinels where fewer are visible, after the window rows: the kernel wants the
valid
rows as one prefix), the window rows are a constant once the window is full, and the step's compressed row and
index key go to the cache's last row (never reached by a decodable position;
visible through the mask) and are copied to row ``start_pos // ratio`` by two small ops after the step. The
per-step
host inputs (token, RoPE rows, mask rows, Engram lookups) are copied into fixed device buffers.

Tracing (``trace=True``). Steps run eagerly until ``warmup`` of them compiled the kernels and settled the
persistent
resources, then the next even position captures the even graph (ratio-2 compressors keep the token's
projections as
the carry) and the following odd position the odd graph (they pool and write the group's row); the captured
step is
replayed at once for its result and every later step replays its parity's capture with the host inputs
refreshed. The
MoE's sub-device switches split each capture (``SubDeviceTraceController``), as in ``V41PrefillTrace``.

State reuse: the window region of each KV tensor takes the layer's carry at rows [0, 128) and the token's row
at 128,
so ``update_window_carry(layer, 1)`` is the ring shift; ``compressor_carry`` is consumed.

Limits (asserted): text only; positions from ``SLIDING_WINDOW - 1`` to below ``max_seq_len - 32``; compressed
rows up
to the candidate source's block budget (``CANDIDATE_TOPK_BLOCKS * CANDIDATE_BLOCK_SIZE`` rows: shorter
contexts
select over the full row, as the reference).
"""

import os
import time
from dataclasses import dataclass, field
from unittest import mock

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import mhc_expand
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.tt.v41.attention import TOPK_ALIGN, TtV41Attention
from models.demos.deepseek_v3_d_p.tt.v41.cache import WINDOW_SLOT, V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.head_layout import o_heads, q_heads
from models.demos.deepseek_v3_d_p.tt.v41.indexer import TtV41Indexer
from models.demos.deepseek_v3_d_p.tt.v41.layout import TP_AXIS
from models.demos.deepseek_v3_d_p.tt.v41.moe import TtV41Moe
from models.demos.deepseek_v3_d_p.tt.v41.moe_shared import SHARED_IN_FFN, install_shared_in_ffn
from models.demos.deepseek_v3_d_p.tt.v41.moe_tail import install_decode_tail
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp4_e4m3_qdq, fp8_qdq
from models.demos.deepseek_v3_d_p.tt.v41.rope import cos_sin
from models.demos.deepseek_v3_d_p.tt.v41.routing import install_decode_routing
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

ROWS = 32  # the decode tile: the token on every row (every fused V4.1 op works in whole tiles)
# the decode routing (tt/v41/routing.py) runs the MoE on the tile itself; the prefill routing setup
# (masked_bincount)
# wants a multiple of 64 tokens per chip, so the tile is repeated for it
DECODE_ROUTING = os.environ.get("V41_DECODE_ROUTING", "0") == "1"
# V41_DECODE_ROUTING_ROWS=64 keeps the repeated 64-row tile with the decode routing (a bisection knob)
MOE_ROWS = int(os.environ.get("V41_DECODE_ROUTING_ROWS", ROWS)) if DECODE_ROUTING else 64
# the decode MoE tail (tt/v41/moe_tail.py) sums the combine buffer's slots on 80 cores; it takes the 32-row
# tile
DECODE_TAIL = os.environ.get("V41_DECODE_TAIL", "0") == "1"
assert not DECODE_TAIL or DECODE_ROUTING, "V41_DECODE_TAIL=1 needs V41_DECODE_ROUTING=1 (the 32-row MoE tile)"
# the decode shared expert's partial summed before the tail's reduce-scatter instead of reduce-scattered
# itself
# (tt/v41/moe_shared.py): ``rs`` with today's matmuls, ``ffn`` as one routed-expert FFN program, whose weights
# V41_SHARED_IN_FFN_DTYPE=bfp4 copies to bfp4 (bfp8: the tensors as they are)
assert SHARED_IN_FFN == "0" or DECODE_TAIL, "V41_SHARED_IN_FFN needs V41_DECODE_TAIL=1 (the decode tail's sum)"
SHARED_IN_FFN_DTYPE = {"bfp8": None, "bfp4": ttnn.bfloat4_b}[os.environ.get("V41_SHARED_IN_FFN_DTYPE", "bfp8")]


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


@dataclass
class _Trace:
    """One position parity's capture: its segmented trace, its device outputs (read after each replay)."""

    controller: SubDeviceTraceController
    logits: ttnn.Tensor
    sp_row: int
    blocks: list  # (layer, streams) device tensors, kept allocated for ``on_block`` after a replay


@dataclass
class _Step:
    """One step's host facts: position, parity, the compressed rows it writes (source layer -> row)."""

    p: int
    writes: dict = field(default_factory=dict)


class TtV41Decoder:
    def __init__(
        self,
        model: TtV41Transformer,
        weight_cache_path,
        routed_expert_weights_dtype=None,
        trace: bool = False,
        warmup: int = 4,
    ):
        """``model``: the built prefill transformer whose blocks, head, Engram modules and states are reused.
        ``weight_cache_path``: the device MoE cache the model was built from (the decode MoEs load their gate
        and
        shared expert from it; the routed experts are the prefill blocks' tensors). ``trace``: capture the
        step per
        position parity after ``warmup`` eager steps and replay it (the device needs a trace region)."""
        assert weight_cache_path is not None, "the decode MoE is built from the device weight cache"
        self.model, self.mesh_device, self.config = model, model.mesh_device, model.config
        if routed_expert_weights_dtype is None:
            routed_expert_weights_dtype = model.routed_expert_weights_dtype  # the decode MoEs share its experts
        self.sp, self.tp = tuple(self.mesh_device.shape)
        # a model with DSpark seeding decodes as any other: the drafter (tt/v41/dspark_drafter.py) is a
        # separate
        # module with its own ring writes; the step itself reads nothing of it
        pre = torch.zeros(1, 1, ROWS, self.config.HC_MULT)
        pre[..., 0] = 1.0
        self.pre_mix = self._replicated(pre, ttnn.float32)
        self.moes = [
            self._decode_moe(block, layer, weight_cache_path, routed_expert_weights_dtype)
            for layer, block in zip(model.layers, model.blocks)
        ]
        self.on_block = None  # optional callback (layer, streams) after each block, for acceptance checks
        self.trace, self.warmup = trace, warmup
        self.buffers = {}  # the fixed device input buffers, allocated by the first step
        self.window_rows = None  # sparse_sdpa index rows of the (full) window, a constant
        # per state (keyed by id; a capture reads and writes that state's cache tensors): its captures by
        # parity,
        # how many eager steps it has run, and its compressor carry buffers (``prepare``); the Engram hash
        # history
        # is the state's ``engram_history``
        self.traces: dict[tuple[int, int], _Trace] = {}
        self.eager_steps: dict[int, int] = {}
        self.carries: dict[int, dict] = {}
        self.last = {}  # how the last step ran: mode (eager / capture / replay), device_s for replays
        # V41_DECODE_INPUT_TIMING=1: ``last["items"]`` holds the host inputs' and the staging's time per item
        # (ms)
        self.input_timing = os.environ.get("V41_DECODE_INPUT_TIMING", "0") == "1"
        self.items: dict[str, float] = {}
        # V41_DECODE_STAGE_PROBE=1: a 128-byte copy before the step's staging, timed as
        # ``stage.probe``: the served step's first host-to-device copy after the replay costs 1.7 to 2.0 ms
        # against
        # 0.1 in isolation; the probe says whether the cost is the first copy's whatever its size
        self.stage_probe = os.environ.get("V41_DECODE_STAGE_PROBE", "0")  # "1": before the staging; "pre": after
        # V41_DECODE_STAGE_BATCH=1: the step's host-to-device copies (the packed inputs, the
        # Engram rows) issued as one ``ttnn.copy_host_to_device_tensors`` (one dispatch round: one wake-up of
        # the
        # 32 per-device workers instead of one a copy)
        self.stage_batch = os.environ.get("V41_DECODE_STAGE_BATCH", "0") == "1"
        self.probe: tuple | None = None  # the replay's issue, before its sync (``issue_s`` then carries it)

    # --- construction ----------------------------------------------------------------------------------
    def _replicated(self, host: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT, device=True) -> ttnn.Tensor:
        return ttnn.from_torch(
            host,
            device=self.mesh_device if device else None,
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _decode_moe(self, block, layer: int, root, dtype) -> TtV41Moe:
        """A ``TtV41Moe`` for one 32-row tile per chip (``seq_len = ROWS * sp``), its routed experts the
        prefill
        block's device tensors (``TtRoutedExpert`` loads them from the cache otherwise: a second copy per
        layer),
        shared expert and dispatch in sequence (the overlap's sub-device switches would split the step's trace
        into
        81 segments). Both are injected under ``mock.patch``: ``TtV41Moe.__init__``'s source is part of the
        weight
        cache's key (``tests/v41/weight_cache.py``), so it cannot grow a parameter without invalidating the
        cache."""
        source = block.ffn.moe.routed_expert
        shared = (source.gate_projs, source.up_projs, source.down_projs)
        build_moe = TtPrefillBlock._build_moe

        def build_sequential(*args, **kwargs):
            return build_moe(*args, **{**kwargs, "overlap_shared_expert_with_dispatch": False})

        with (
            mock.patch.object(
                TtRoutedExpert, "_convert_and_cache_expert_weights", staticmethod(lambda *a, **k: shared)
            ),
            mock.patch.object(TtPrefillBlock, "_build_moe", staticmethod(build_sequential)),
        ):
            moe = TtV41Moe(
                self.mesh_device,
                self.config,
                layer,
                {},
                MOE_ROWS * self.sp,
                routed_expert_weights_dtype=dtype,
                weight_cache_path=root,
            )
        # the shared expert's reduce-scatter intermediate is one buffer per mesh
        # (TtCCL.get_shared_rs_intermediate),
        # allocated by the prefill MoE for its Linear topology (forced under the overlap); the Ring path would
        # want
        # one of the decode input's shape, so the decode shared expert keeps Linear
        moe.moe.shared_expert.topology = ttnn.Topology.Linear
        if DECODE_ROUTING:
            install_decode_routing(moe, self.moe_real_rows())
        if DECODE_TAIL:
            install_decode_tail(moe, ccl=block.attn.ccl)
        if SHARED_IN_FFN != "0":
            install_shared_in_ffn(moe, SHARED_IN_FFN, SHARED_IN_FFN_DTYPE)
        return moe

    def moe_real_rows(self) -> int | None:
        """The real rows of the MoE tile under a padding config: every row here (the token is on every row)."""
        return None

    # --- host inputs -> fixed device buffers
    # -----------------------------------------------------------------
    def _host_inputs(self, token: int, state: V41PrefillState, step: _Step) -> dict:
        """The step's host tensors (replicated, device layouts), keyed as the device buffers."""
        p, cfg, g = step.p, self.config, state.geometry
        ids = torch.full((1, 1, ROWS), token, dtype=torch.int32)
        inputs = {"token": self._replicated(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device=False)}

        def rope(compressed: bool, position: int):
            positions = torch.full((ROWS,), position, dtype=torch.int64)
            return tuple(self._replicated(t, ttnn.bfloat16, device=False) for t in cos_sin(cfg, compressed, positions))

        inputs["rope0"], inputs["ropec"] = rope(False, p), rope(True, p)
        inputs["ropeg"] = rope(True, p // 2 * 2)  # a ratio-2 group's first position
        for ratio in (1, 2):
            cap = g.compressed_rows(ratio)
            visible = (p + 1) // ratio
            written = visible - 1 if p % ratio == ratio - 1 else visible  # the step's own row sits in the scratch
            mask = torch.zeros(1, 1, 1, cap)
            mask[..., written:] = float("-inf")
            if written < visible:
                mask[..., cap - 1] = 0.0
            inputs[f"mask{ratio}"] = self._replicated(mask, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device=False)
        if self.model.engram:
            hash_ = self.model.engram_hash
            ids, _ = hash_(torch.tensor([token]), state.engram_history)  # [1, n_layers, 24]
            chunk = ROWS * self.sp  # prepare_host splits chunk * 24 lookups over the chips in tiles
            inputs["engram"] = {
                l: e.prepare_host(ids[:, hash_.layer_index(l)].expand(chunk, -1).contiguous(), chunk)
                for l, e in self.model.engram.items()
            }
        return inputs

    def _stage(self, inputs: dict) -> None:
        """Copy the host inputs into the fixed device buffers (allocated by the first step)."""
        if not self.buffers:
            for key, value in inputs.items():
                if key == "engram":
                    self.buffers[key] = {l: e.to_device(self.mesh_device) for l, e in value.items()}
                elif isinstance(value, tuple):
                    self.buffers[key] = tuple(ttnn.to_device(t, self.mesh_device) for t in value)
                else:
                    self.buffers[key] = ttnn.to_device(value, self.mesh_device)
            return
        tick = self._tick("stage")
        if self.stage_probe == "1":
            self.probe_copy()
            tick("probe")
        if self.stage_batch:
            hosts, devs = [], []
            for key, value in inputs.items():
                if key == "engram":
                    for l, e in value.items():
                        for h, d in e.copy_pairs(self.buffers[key][l]):
                            hosts.append(h)
                            devs.append(d)
                elif isinstance(value, tuple):
                    for host, dev in zip(value, self.buffers[key]):
                        hosts.append(host)
                        devs.append(dev)
                else:
                    hosts.append(value)
                    devs.append(self.buffers[key])
            ttnn.copy_host_to_device_tensors(hosts, devs)
            tick("batch")
            return
        for key, value in inputs.items():
            if key == "engram":
                for l, e in value.items():
                    e.copy_into(self.buffers[key][l])
            elif isinstance(value, tuple):
                for host, dev in zip(value, self.buffers[key]):
                    ttnn.copy_host_to_device_tensor(host, dev)
            else:
                ttnn.copy_host_to_device_tensor(value, self.buffers[key])
            tick(key)

    def probe_copy(self) -> None:
        """One 128-byte host-to-device copy (``V41_DECODE_STAGE_PROBE``)."""
        if self.probe is None:
            host = self._replicated(
                torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device=False
            )
            self.probe = (host, ttnn.to_device(host, self.mesh_device))
        ttnn.copy_host_to_device_tensor(*self.probe)

    def _tick(self, prefix: str, items: dict | None = None):
        """A timer: ``tick(name)`` adds the time since the last tick to ``items[prefix.name]`` (ms; ``items``
        default ``self.items``) when ``V41_DECODE_INPUT_TIMING`` is set; a no-op otherwise."""
        if not self.input_timing:
            return lambda name: None
        last = [time.perf_counter()]

        def tick(name):
            now = time.perf_counter()
            d = self.items if items is None else items
            d[f"{prefix}.{name}"] = round(d.get(f"{prefix}.{name}", 0.0) + 1e3 * (now - last[0]), 3)
            last[0] = now

        return tick

    def _window_rows(self) -> ttnn.Tensor:
        """``sparse_sdpa`` index rows ``[1, 1, ROWS, SLIDING_WINDOW]`` uint32 row-major of the window rows of
        positions ``p - 127 .. p`` (row 128 - j holds position p - j): a constant, the window being full."""
        if self.window_rows is None:
            window = [WINDOW_SLOT - j for j in range(self.config.SLIDING_WINDOW - 1, -1, -1)]
            host = torch.tensor(window, dtype=torch.int32).expand(1, 1, ROWS, -1).contiguous()
            self.window_rows = self._replicated(host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        return self.window_rows

    # --- one step ----------------------------------------------------------------------------------------
    def prepare(self, state: V41PrefillState) -> None:
        """Allocate ``state``'s decode buffers: the compressor carry of every ratio-2 source as a fixed tensor
        the
        even step writes (in the graph, ``ttnn.copy``) and the odd step reads. Every state a decoder will
        serve must
        be prepared before its first capture: a tensor allocated after a capture may lie on that capture's
        freed
        intermediates, and a replay then overwrites it (a second slot's carry under the first slot's
        replays). ``decode`` prepares a state on its first step when nothing is captured yet."""
        key = id(state)
        if key in self.carries:
            return
        assert not self.traces, "every state must be prepared before the first capture (see prepare's docstring)"
        d = self.config.HEAD_DIM
        self.carries[key] = {
            layer: tuple(self._replicated(torch.zeros(1, 1, ROWS, d), ttnn.float32) for _ in range(2))
            for layer in state.compressor_carry
        }

    def begin(self, tokens: torch.Tensor, state: V41PrefillState) -> None:
        """Start decoding ``state`` after ``prefill(tokens)`` filled it: seeds its Engram hash history (the
        last
        compressed ids) and moves the prefill's compressor carry (an odd prompt) into the state's carry
        buffers."""
        if self.model.engram:
            hash_ = self.model.engram_hash
            n = hash_.layout.max_ngram_size - 1
            state.engram_history = hash_.token_map[tokens[-n:].to(torch.int64)].clone()
        self.prepare(state)
        for layer, carry in state.compressor_carry.items():
            if carry is None:
                continue
            for src, dst in zip(carry, self.carries[id(state)][layer]):
                ttnn.copy(self._rows(src), dst)
            state.compressor_carry[layer] = self.carries[id(state)][layer]

    def decode(self, token: int, state: V41PrefillState) -> torch.Tensor:
        """Append ``token`` at position ``state.start`` -> fp32 logits ``[vocab]`` of the next position; the
        state
        advances by one."""
        p = state.start
        g = state.geometry
        assert p + ROWS <= g.max_seq_len, f"position {p} beyond the state's {g.max_seq_len}"
        assert p >= self.config.SLIDING_WINDOW - 1, f"position {p}: decode starts after a full window"
        step = _Step(p)
        for source in state.kv:
            ratio = self.config.compress_ratio(source)
            if p % ratio == ratio - 1:
                step.writes[source] = p // ratio
        t0 = time.time()
        inputs = self._host_inputs(token, state, step)
        t1 = time.time()
        self._stage(inputs)
        t2 = time.time()
        key, parity = id(state), p % 2
        self.prepare(state)
        if not self.trace:
            mode = "eager"
        elif (key, parity) in self.traces:
            mode = "replay"
        elif self.eager_steps.get(key, 0) >= self.warmup and (parity == 0 or (key, 0) in self.traces):
            mode = "capture"
        else:
            mode = "eager"
        self.last = {"mode": mode, "inputs_s": round(t1 - t0, 4), "stage_s": round(t2 - t1, 4)}
        if mode == "eager":
            self.eager_steps[key] = self.eager_steps.get(key, 0) + 1
            logits, (sp_row, _) = self._forward(state, step, self.on_block)
            self._finish(state, step)
            out = self.model.head._tile_to_host(logits, sp_row)[0]
            ttnn.deallocate(logits)
        else:
            if mode == "capture":
                self.traces[(key, parity)] = self._capture(state, step)
            trace = self.traces[(key, parity)]
            ttnn.synchronize_device(self.mesh_device)  # the staged inputs have landed: time the step alone
            t3 = time.time()
            trace.controller.replay(blocking=False)
            self._finish(state, step)
            t4 = time.time()
            ttnn.synchronize_device(self.mesh_device)
            t5 = time.time()
            out = self.model.head._tile_to_host(trace.logits, trace.sp_row)[0]
            # device_s: replay issue to completion (the device's time per token plus the row copies); issue_s:
            # the
            # host's share of it (enqueuing the replay and the eager row copies); sync_s: waiting for the
            # staging
            self.last.update(
                sync_s=round(t3 - t2, 4),
                device_s=round(t5 - t3, 4),
                issue_s=round(t4 - t3, 4),
                read_s=round(time.time() - t5, 4),
            )
            if self.on_block is not None:
                for layer, x in trace.blocks:
                    self.on_block(layer, x)
        t6 = time.time()
        if self.model.engram:
            self._advance_history(token, state)
        state.advance(1)
        self.last["history_s"] = round(time.time() - t6, 4)
        return out

    def replay_device_time(self, repeats: int = 3) -> dict:
        """Device time of one replay per captured parity without host work in between (the step's device cost;
        replays the captured graphs on the state's current inputs, which advances nothing but rewrites the
        caches' scratch rows and the window carries: call it when done decoding) -> parity -> min seconds."""
        times = {}
        for (_, parity), trace in self.traces.items():
            ttnn.synchronize_device(self.mesh_device)
            best = None
            for _ in range(repeats):
                t0 = time.time()
                trace.controller.replay(blocking=False)
                ttnn.synchronize_device(self.mesh_device)
                dt = time.time() - t0
                best = dt if best is None else min(best, dt)
            times[parity] = round(best, 4)
        return times

    def _capture(self, state: V41PrefillState, step: _Step) -> _Trace:
        """Capture the step's device graph (not run: the caller replays it) as one segmented trace."""
        controller = SubDeviceTraceController(self.mesh_device)
        for moe in self.moes:
            moe.set_trace_controller(controller)
        blocks = []
        ttnn.synchronize_device(self.mesh_device)
        controller.begin_capture()
        try:
            logits, (sp_row, _) = self._forward(state, step, lambda layer, x: blocks.append((layer, x)))
        finally:
            controller.end_capture()
            for moe in self.moes:
                moe.set_trace_controller(None)
        return _Trace(controller, logits, sp_row, blocks)

    def _finish(self, state: V41PrefillState, step: _Step) -> None:
        """After the step's graph: move the compressed rows it wrote from the scratch row to their place."""
        for source, j in step.writes.items():
            cap = state.geometry.compressed_rows(self.config.compress_ratio(source))
            state.copy_compressed_row(source, cap - 1, j)

    def traced(self, state: V41PrefillState) -> bool:
        """Whether both parities of ``state``'s step are captured."""
        return all((id(state), parity) in self.traces for parity in (0, 1))

    def release(self) -> None:
        for trace in self.traces.values():
            trace.controller.release()
        self.traces = {}

    @property
    def trace_segments(self) -> int:
        return sum(t.controller.num_segments for t in self.traces.values())

    def _forward(self, state: V41PrefillState, step: _Step, on_block):
        """The step's device graph on the staged buffers -> (the next position's logits row on device,
        (sp_row,
        offset)); the state's caches and carries are updated in place."""
        model = self.model
        # the mask rows on every row of the tile (an H-broadcast add masked only row 0; the query is the last
        # row)
        masks = {
            ratio: ttnn.to_layout(ttnn.concat([self.buffers[f"mask{ratio}"]] * ROWS, dim=2), ttnn.TILE_LAYOUT)
            for ratio in (1, 2)
        }
        h = model.embedding(self.buffers["token"])
        x = mhc_expand(ttnn.typecast(h, ttnn.float32), self.config.HC_MULT)
        pre = self.pre_mix
        for layer, block, moe in zip(model.layers, model.blocks, self.moes):
            if layer in model.engram:
                x = model.engram[layer](x, self.buffers["engram"][layer])
            x, pre = block.residual(
                x,
                pre,
                attention=lambda h, a=block.attn, n=block.attn_norm: self._attention(a, n(h), state, step, masks),
                ffn=lambda h, m=moe, n=block.ffn_norm: self._ffn(m, n(h)),
            )
            if on_block is not None:
                on_block(layer, x)
        final = ttnn.typecast(model.blocks[-1].residual.final_collapse(x, pre), ttnn.bfloat16)
        logits, (sp_row, offset) = model.head(final, 0)
        return model.head.scored_rows(logits, offset, 1), (sp_row, 0)

    def _ffn(self, moe: TtV41Moe, h):
        """The decode MoE on the tile repeated to MOE_ROWS rows (identical rows route identically) -> ROWS
        rows."""
        if MOE_ROWS == ROWS:
            return moe(h)
        out = moe(ttnn.concat([h] * (MOE_ROWS // ROWS), dim=2))
        return ttnn.slice(out, [0, 0, 0, 0], [1, 1, ROWS, out.shape[3]])

    # --- Engram -------------------------------------------------------------------------------------------
    def _advance_history(self, token: int, state: V41PrefillState) -> None:
        hash_ = self.model.engram_hash
        _, state.engram_history = hash_(torch.tensor([token]), state.engram_history)

    # --- attention ----------------------------------------------------------------------------------------
    def _attention(self, attn: TtV41Attention, x, state: V41PrefillState, step: _Step, masks: dict):
        """``TtV41Attention.forward`` for one token (``x`` [1, 1, ROWS, hidden/tp] bf16, the token on every
        row);
        ``masks``: per ratio the additive index mask ``[1, 1, ROWS, W]`` of the step."""
        p, layer = step.p, attn.layer
        cos, sin = self.buffers["ropec" if attn.ratio else "rope0"]
        xq = fp8_qdq(x)
        qr = ttnn.rms_norm(attn.ccl.tp_all_reduce(attn._dense(xq, "wq_a")), weight=attn.q_norm, epsilon=attn.eps)
        q = attn._dense(fp8_qdq(qr), "wq_b")  # [1, 1, ROWS, H/tp * head_dim]
        kv = ttnn.rms_norm(attn.ccl.tp_all_reduce(attn._dense(xq, "wkv")), weight=attn.kv_norm, epsilon=attn.eps)
        kv_format = state.layer_kv_format(layer)
        reference_values = kv_format == MlaKvCacheFormat.BF16_RM
        kv = attn._rope(kv, cos, sin)
        kv_tensor = state.write_window_rows(layer, fp8_qdq(kv) if reference_values else kv)

        compressed = None
        if attn.ratio:
            ratio = attn.ratio
            if attn.is_kv_source:
                latent = self._compress(attn.compressor, x, state, layer, p)
                if latent is not None:
                    c_rope = self.buffers["ropeg" if ratio == 2 else "ropec"]  # the group's first position
                    keys = attn.index_keys(latent, c_rope)
                    comp_kv = attn._rope(latent, *c_rope)
                    scratch = state.geometry.compressed_rows(ratio) - 1
                    assert step.writes.get(layer) == p // ratio, (layer, p)
                    state.write_compressed_row(
                        layer, fp4_e4m3_qdq(comp_kv) if reference_values else comp_kv, keys, scratch
                    )
            if attn.is_index_source:
                src = self.config.kv_source(layer)
                compressed, candidates = self._index(
                    attn.indexer,
                    x,
                    qr,
                    state.index_k[src],
                    masks[ratio],
                    state.selection.get("candidates"),
                )
                state.selection["topk"] = compressed
                if candidates is not None:
                    state.selection["candidates"] = candidates
            else:
                compressed = state.selection["topk"]
        rows = self._index_rows(compressed, state.geometry.window_rows)

        q = attn.ccl.tp_all_gather(q, dim=3)  # all heads, in head order (wq_b is column-parallel over heads)
        out = ttnn.transformer.sparse_sdpa(
            q_heads(q, cos, sin, attn.trans_mat, attn.heads, attn.rope_dim),
            kv_tensor,
            rows,
            attn.head_dim,
            kv_format=kv_format.sparse_sdpa_format,
            scale=attn.scale,
            k_chunk_size=next(c for c in (128, 64, 32) if rows.shape[-1] % c == 0),
            attention_sink=attn.sink,
        )
        out = o_heads(out, cos, ttnn.neg(sin), attn.trans_mat, attn.o_groups, attn.rope_dim)  # [1, G, ROWS, ...]
        if self.tp > 1:
            out = ttnn.mesh_partition(out, dim=1, cluster_axis=TP_AXIS)  # this chip's groups
        state.update_window_carry(layer, 1)
        return attn._o_proj(out)

    def _compress(self, compressor, x, state: V41PrefillState, layer: int, p: int):
        """The reference ``Compressor`` at ``start_pos = p``: the latent of the group ``p`` completes, else
        None
        (the token's fp32 projections kept as the carry)."""
        if compressor.ratio == 1:
            kv = compressor._project(x, compressor.wkv, ttnn.bfloat16)
            return ttnn.rms_norm(kv, weight=compressor.norm, epsilon=compressor.eps)
        xf = ttnn.typecast(x, ttnn.float32)
        kv = compressor._project(xf, compressor.wkv, ttnn.float32)
        score = compressor._project(xf, compressor.wgate, ttnn.float32)
        if (p + 1) % compressor.ratio:
            buffers = self.carries[id(state)][layer]
            for src, dst in zip((kv, score), buffers):
                ttnn.copy(src, dst)
            state.compressor_carry[layer] = buffers
            return None
        carry = state.compressor_carry[layer]
        assert carry is not None, f"layer {layer}: no carried projections for position {p - 1}"
        k0, s0 = (self._rows(t) for t in carry)  # the buffers, or a prefill's row before begin() moved it
        state.compressor_carry[layer] = None
        m = ttnn.maximum(s0, score)
        e0, e1 = ttnn.exp(ttnn.subtract(s0, m)), ttnn.exp(ttnn.subtract(score, m))
        pooled = ttnn.divide(ttnn.add(ttnn.multiply(k0, e0), ttnn.multiply(kv, e1)), ttnn.add(e0, e1))
        return ttnn.rms_norm(ttnn.typecast(pooled, ttnn.bfloat16), weight=compressor.norm, epsilon=compressor.eps)

    def _rows(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """A prefill carry row ``[1, 1, 1, d]`` -> ``[1, 1, ROWS, d]`` (the row on every row); decode carries
        pass."""
        if t.shape[2] == ROWS:
            return t
        assert t.shape[2] == 1, tuple(t.shape)
        row = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.to_layout(ttnn.concat([row] * ROWS, dim=2), ttnn.TILE_LAYOUT)

    def _index(self, indexer: TtV41Indexer, x, qr, index_k, mask, candidates):
        """The indexer for one query over the whole index-K cache, ``mask`` ``[1, 1, ROWS, W]`` (-inf at the
        rows not
        yet written) -> (top-k rows ``[1, 1, ROWS, INDEX_TOPK]`` uint32 row-major, the same selection on every
        row,
        sentinels past the visible rows; published candidates or None)."""
        cfg = self.config
        width = index_k.shape[2]
        assert (
            width // cfg.CANDIDATE_BLOCK_SIZE <= cfg.CANDIDATE_TOPK_BLOCKS
        ), f"decode over {width} compressed rows needs the candidate-block selection (not implemented)"
        cos, sin = self.buffers["ropec"]
        q = ttnn.linear(fp8_qdq(qr), indexer.wq_b, compute_kernel_config=indexer.compute_kernel_config)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=indexer.heads, num_kv_heads=0, transpose_k_heads=False
        )
        q = indexer.qdq(indexer.rope(q, cos, sin))
        w = indexer.ccl.tp_all_reduce(ttnn.linear(x, indexer.weights_proj))
        k = ttnn.to_layout(index_k, ttnn.TILE_LAYOUT)
        # the kernel's causal mask is t <= chunk_start + s (+ a per-chip offset): with chunk_start = width -
        # 32 the
        # last row sees every column on every chip; the unwritten columns are masked here (zero keys score 0)
        score = indexer._score(q, k, w, chunk_start_idx=width - 32)  # [1, 1, ROWS, width] bf16 row-major
        score = ttnn.to_layout(ttnn.add(ttnn.to_layout(score, ttnn.TILE_LAYOUT), mask), ttnn.ROW_MAJOR_LAYOUT)
        topk, published = indexer.select(score, None, 0, cfg.INDEX_TOPK, candidates)
        last = ttnn.slice(topk, [0, 0, ROWS - 1, 0], [1, 1, ROWS, topk.shape[3]])
        return ttnn.concat([last] * ROWS, dim=2), published

    def _index_rows(self, compressed, window_base: int) -> ttnn.Tensor:
        """``sparse_sdpa`` index rows ``[1, 1, ROWS, K]`` uint32 row-major: the window rows, then the selected
        compressed rows with their sentinel tail (none for a ratio-0 layer): the kernel wants the valid rows
        as one
        prefix, and the window is full."""
        window = self._window_rows()
        if compressed is None:
            return window
        comp = ttnn.typecast(ttnn.to_layout(compressed, ttnn.TILE_LAYOUT), ttnn.int32)
        comp = ttnn.where(ttnn.eq(comp, -1), -1, ttnn.add(comp, window_base))
        comp = ttnn.to_layout(ttnn.typecast(comp, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        rows = ttnn.concat([window, comp], dim=-1)
        assert rows.shape[3] % TOPK_ALIGN == 0, tuple(rows.shape)
        return rows
