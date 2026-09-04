# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 embedding performance benchmark.

test_perf captures the encoder trace and times the trace replay. pytest reads
the hardware while it collects the tests and selects the shape that the local
card supports:

  * a 2-chip Wormhole card runs batch 12 and sequence length 8192 across both
    chips, with fabric enabled
  * any other card runs the batch sweep at sequence length 512 on one device

Run it from the tt-metal root:

    TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/perf.py::test_perf -s
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

NUM_ITERATIONS = 10


def detected_n300() -> bool:
    """True when the visible hardware is a 2-chip Wormhole card.

    pytest calls this while it collects the tests, so it must not open a
    device. ttnn.get_num_devices() reports the count that TT_VISIBLE_DEVICES
    exposes.
    """
    return ttnn.get_arch_name() == "wormhole_b0" and ttnn.get_num_devices() == 2


def report_perf(batch_size, seq_len, valid_len, masked, best_ms, avg_ms):
    """Log one result block."""
    total_tokens = batch_size * valid_len
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  BGE-M3  B{batch_size} S{seq_len}  ({'masked' if masked else 'nomask'})")
    logger.info("=" * 60)
    logger.info(f"  Batch size:           {batch_size}")
    if masked:
        logger.info(f"  Seq length:           {valid_len}")
        logger.info(f"  Seq length (padded):  {seq_len}")
    else:
        logger.info(f"  Seq length:           {seq_len}")
    logger.info(f"  Valid tokens/seq:     {valid_len}")
    logger.info(f"  Total valid tokens:   {total_tokens}")
    logger.info(f"  Iterations:           {NUM_ITERATIONS}")
    logger.info("-" * 60)
    logger.info(f"  Avg latency:          {avg_ms:.3f} ms")
    logger.info(f"  Best latency:         {best_ms:.3f} ms")
    logger.info(f"  Avg embeddings/s:     {batch_size / (avg_ms / 1000):.1f}")
    logger.info(f"  Best embeddings/s:    {batch_size / (best_ms / 1000):.1f}")
    logger.info(f"  Avg tokens/s:         {total_tokens / (avg_ms / 1000):.0f}")
    logger.info(f"  Best tokens/s:        {total_tokens / (best_ms / 1000):.0f}")
    logger.info(f"  Avg requests/s:       {1.0 / (avg_ms / 1000):.3f}")
    logger.info(f"  Best requests/s:      {1.0 / (best_ms / 1000):.3f}")
    logger.info("=" * 60)


def _n300_dp_batchshard(torch_inputs, mesh_device, *, on_device):
    """Shard input_ids / token_type / position on the batch dim across the mesh."""
    mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    kwargs = {"mesh_mapper": mapper}
    if on_device:
        kwargs.update(device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def convert(tensor):
        return ttnn.from_torch(tensor.int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, **kwargs)

    tensors = {
        "input_ids": convert(torch_inputs["input_ids"]),
        "token_type_ids": convert(torch_inputs["token_type_ids"]),
        "position_ids": convert(torch_inputs["position_ids"]),
    }
    if torch_inputs.get("valid_lengths") is not None:
        tensors["attention_mask"] = convert(torch_inputs["valid_lengths"])
    return tensors


def _n300_dp_inputs(pad_token_id, batch, valid_len, seq_len=8192):
    """Build B x seq_len inputs. valid_len < seq_len -> pad the tail and pass a
    compact [B, 1] valid-length mask; valid_len == seq_len -> no mask."""
    input_ids = torch.full((batch, seq_len), pad_token_id, dtype=torch.long)
    input_ids[:, :valid_len] = torch.randint(1, 1000, (batch, valid_len), dtype=torch.long)
    token_type_ids = torch.zeros(batch, seq_len, dtype=torch.long)
    nonpad = (input_ids != pad_token_id).to(torch.int64)
    position_ids = torch.cumsum(nonpad, dim=1) * nonpad + pad_token_id
    valid_lengths = None
    if valid_len < seq_len:
        valid_lengths = torch.full((batch, 1), valid_len, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "position_ids": position_ids,
        "valid_lengths": valid_lengths,
    }


@pytest.mark.parametrize(
    "mesh_device",
    [(2, 1)] if detected_n300() else [1],
    indirect=True,
    ids=["n300_dp2"] if detected_n300() else ["single"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 50_000_000,
            "num_command_queues": 1,
            **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if detected_n300() else {}),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [(12, 8192)] if detected_n300() else [(1, 512), (8, 512), (16, 512), (32, 512)],
    ids=(lambda shapes: [f"b{b}_s{s}" for b, s in shapes])(
        [(12, 8192)] if detected_n300() else [(1, 512), (8, 512), (16, 512), (32, 512)]
    ),
)
@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "masked"])
def test_perf(mesh_device, batch_size, seq_len, masked):
    """Report wall-clock trace-replay time for the shape the local card runs.

    masked uses compact valid lengths, which only the data-parallel path
    accepts. Every other shape rejects that mask, so the masked run applies to
    the N300 shape alone.
    """
    data_parallel = detected_n300()
    if masked and not data_parallel:
        pytest.skip("compact valid-length masking needs the data-parallel path")

    args, model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=batch_size,
        max_seq_len=seq_len,
        dtype=ttnn.bfloat8_b,
        data_parallel=data_parallel,
    )

    valid_lengths = [128, 512, 1024, 2048, 4096] if masked else [seq_len]
    for valid_len in valid_lengths:
        inputs = _n300_dp_inputs(args.pad_token_id, batch_size, valid_len, seq_len)
        device_tensors = _n300_dp_batchshard(inputs, mesh_device, on_device=True)

        out = model.forward(**device_tensors)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(out)

        model.capture_trace(**device_tensors, mesh_device=mesh_device, cq_id=0)
        for _ in range(3):
            model.execute_trace(blocking=True)

        times = []
        for _ in range(NUM_ITERATIONS):
            start = time.perf_counter()
            model.execute_trace(blocking=True)
            times.append((time.perf_counter() - start) * 1000.0)
        model.release_trace()

        times.sort()
        avg_ms = sum(times) / len(times)
        report_perf(batch_size, seq_len, valid_len, masked, times[0], avg_ms)
