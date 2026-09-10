# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dump the MoE routing a real prefill run produced, as an `expert_routing_<model>.safetensors`.

`test_dispatch_combine_perf` replays a recorded routing pattern instead of random indices, because
Dispatch/Combine cost is a function of how tokens spread across experts. Until now no generator for
those captures existed in the tree -- the dsv3/kimi26/glm52 files came from a local patch on #51426
that was never committed, so the next model had to reinvent it. This is that generator.

Usage:
    TT_DS_DUMP_ROUTING=/abs/path/expert_routing_<model>.safetensors pytest <a real-token prefill test>

Optional:
    TT_DS_DUMP_CHUNK_TOKENS   tokens in a full chunk (default 5120); the capture is taken from the
                              first chunk whose REAL length fills that window.

Output matches what `init_helpers.load_captured_routing` reads: one int32 key per MoE layer,
`expert_ids_layer_<N>`, holding `dispatch_group_size x seq_len_per_chip x num_experts_per_tok` RAW
GLOBAL expert ids. The per-column remap to [0, experts_per_col) u {sentinel} happens at load time.

Four traps this encodes, each of which yields a plausible, non-erroring, WRONG capture:
  1. Capture from a real-token test. A `..._chunked_no_pcc` leg uses synthetic in-vocab ids by
     design, so its routing is meaningless. Use the padded/golden-trace leg.
  2. Every chunk is padded up to the full window on device, so tensor width cannot tell real from
     padded -- a 1024-token chunk still reads 640 tokens/chip. Only `actual_isl` can, so gate on it.
  3. Identify the SP shards by mesh coordinate, never by value-deduplicating the device tensors:
     two chips can route identically (an all-padding shard does). Equality is used only to VERIFY
     that the TP columns of a row are replicas.
  4. A leg built `kv_only_last_layer=True` never runs the last layer's MoE, so you get N-1 layers.
"""

import atexit
import os

import torch
from loguru import logger

_STORE = {}
_PATH = None
_REGISTERED = False
_chunk_no = -1
_chosen_chunk = None
_last_layer = None
_layout_error = None


def _flush():
    if _layout_error:
        logger.error(f"routing capture: NOT writing -- {_layout_error}")
        return
    if not _STORE:
        logger.warning("routing capture: nothing collected, not writing")
        return
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(_PATH) or ".", exist_ok=True)
    save_file(dict(_STORE), _PATH)
    shape = tuple(next(iter(_STORE.values())).shape)
    logger.warning(f"routing capture: wrote {len(_STORE)} layer keys, each {shape} int32 -> {_PATH}")


def maybe_capture(indices, layer_idx, actual_isl=None):
    """Collect one layer's routing. No-op unless TT_DS_DUMP_ROUTING is set."""
    global _PATH, _REGISTERED, _chunk_no, _chosen_chunk, _last_layer, _layout_error

    path = os.environ.get("TT_DS_DUMP_ROUTING")
    if not path:
        return
    import ttnn

    _PATH = path
    if not _REGISTERED:
        atexit.register(_flush)
        _REGISTERED = True

    # Layers run 0..N-1 within a chunk, so a non-increasing layer_idx means a new chunk started.
    if _last_layer is None or layer_idx <= _last_layer:
        _chunk_no += 1
    _last_layer = layer_idx
    if _chosen_chunk is not None and _chunk_no != _chosen_chunk:
        return

    shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(indices)]
    mats = [s.to(torch.int32).reshape(-1, s.shape[-1]) for s in shards]
    n = len(mats)
    seq_per_chip, topk = mats[0].shape

    # Trap 2: width is the PADDED width, identical for every chunk. actual_isl is the real length.
    chunk_tokens = int(os.environ.get("TT_DS_DUMP_CHUNK_TOKENS", "5120"))
    dispatch_group_size = n // 4 if n == 32 else n
    if seq_per_chip != chunk_tokens // dispatch_group_size:
        return
    if _chosen_chunk is None:
        logger.info(f"routing capture: chunk {_chunk_no} actual_isl={actual_isl} (want {chunk_tokens})")
    if actual_isl is not None and actual_isl != chunk_tokens:
        return
    if _chosen_chunk is None:
        _chosen_chunk = _chunk_no
        logger.warning(
            f"routing capture: locked chunk {_chunk_no} "
            f"(actual_isl={actual_isl}, seq_len_per_chip={seq_per_chip}, topk={topk}, devices={n})"
        )

    # Trap 3: devices come back row-major, so on an 8x4 galaxy row r / col c is index r*4+c. Rows are
    # the SP token-shards; the 4 columns are TP replicas of the same tokens. Verify that, then take
    # one column. A mismatch means the layout assumption is wrong -- refuse to write, but let the
    # model run finish rather than failing someone's test for a debug tool.
    if n == 32:
        rows, cols = 8, 4
        for r in range(rows):
            for c in range(1, cols):
                if not torch.equal(mats[r * cols], mats[r * cols + c]):
                    _layout_error = (
                        f"layer {layer_idx}: mesh row {r} col {c} differs from col 0; the TP columns "
                        f"should be replicas, so the row-major (8,4) assumption is wrong"
                    )
                    logger.error(f"routing capture: {_layout_error}")
                    return
        column = [mats[r * cols] for r in range(rows)]
    elif n == 8:
        column = mats
    else:
        _layout_error = f"layer {layer_idx}: unexpected device count {n}"
        logger.error(f"routing capture: {_layout_error}")
        return

    _STORE[f"expert_ids_layer_{layer_idx}"] = torch.stack(column, dim=0).reshape(len(column), -1)
