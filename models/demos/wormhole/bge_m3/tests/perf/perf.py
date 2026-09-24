# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 embedding performance benchmark.

test_perf captures the encoder trace and times the trace replay. pytest reads
the hardware while it collects the tests and selects the shape that the local
card supports:

  * a 2-chip Wormhole card runs batch 12 and sequence length 8192 across both
    chips, with fabric enabled
  * any other card runs the batch sweep at sequence length 512 on one device

``mode`` selects what the timing loop covers:

  * ``forward``  replays the trace only, so the device time carries no host cost
  * ``2cq``      streams the next input on a second queue while the trace runs
  * ``h2d_d2h``  times one whole request: input upload, trace, output download

The mesh decides whether a mode runs data parallel. On a 2-chip card every mode
shards the batch across both chips, so ``2cq`` there is the data-parallel
two-queue case. One report format serves every mode.

Run it from the tt-metal root:

    TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/perf.py::test_perf -s

Select one mode with -k:

    TT_VISIBLE_DEVICES=0 pytest .../perf.py::test_perf -k "2cq and b32" -s
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

NUM_ITERATIONS = 10

# (mode, num_command_queues). The queue count opens the device, so it pairs with
# the mode instead of multiplying against it.
MODE_PARAMS = [
    ("forward", 1),
    ("2cq", 2),
    ("h2d_d2h", 1),
]


def detected_n300() -> bool:
    """True when the visible hardware is a 2-chip Wormhole card.

    pytest calls this while it collects the tests, so it must not open a
    device. ttnn.get_num_devices() reports the count that TT_VISIBLE_DEVICES
    exposes.
    """
    return ttnn.get_arch_name() == "wormhole_b0" and ttnn.get_num_devices() == 2


def report_perf(batch_size, seq_len, valid_len, masked, best_ms, avg_ms, mode="forward"):
    """Log one result block."""
    total_tokens = batch_size * valid_len
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  BGE-M3  B{batch_size} S{seq_len}  ({'masked' if masked else 'nomask'})  mode={mode}")
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
    logger.info(f"  Mode:                 {mode}")
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


def _stage_on_host(torch_inputs, mesh_device):
    """Build host tensors that carry the mesh shard layout.

    ``device=None`` keeps them on host. ``copy_host_to_device_tensor`` then
    streams each shard onto its own chip without running the mapper again, which
    is how a captured trace receives new input.
    """
    return _n300_dp_batchshard(torch_inputs, mesh_device, on_device=False)


def _refill_device(host_tensors, device_tensors, *, cq_id=0):
    """Overwrite the trace input slots with new host data."""
    for key, host in host_tensors.items():
        ttnn.copy_host_to_device_tensor(host, device_tensors[key], cq_id=cq_id)


def _time_forward(model, mesh_device, device_tensors, host_tensors):
    """Replay the trace only. Reports device time without host transfer."""
    times = []
    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        model.execute_trace(blocking=True)
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def _time_2cq(model, mesh_device, device_tensors, host_tensors):
    """Upload the next input on queue 1 while queue 0 replays the trace.

    The queues overlap, so one iteration cannot be timed alone. Time the whole
    run and divide, and report the amortized figure as both best and average.
    """
    write_event = ttnn.record_event(mesh_device, 1)

    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS - 1):
        ttnn.wait_for_event(0, write_event)
        op_event = ttnn.record_event(mesh_device, 0)
        model.execute_trace(blocking=False, synchronize=False)

        ttnn.wait_for_event(1, op_event)
        _refill_device(host_tensors, device_tensors, cq_id=1)
        write_event = ttnn.record_event(mesh_device, 1)

    ttnn.wait_for_event(0, write_event)
    model.execute_trace(blocking=False, synchronize=False)
    ttnn.synchronize_device(mesh_device)
    amortized_ms = (time.perf_counter() - start) * 1000.0 / NUM_ITERATIONS
    return [amortized_ms] * NUM_ITERATIONS


def _time_h2d_d2h(model, mesh_device, device_tensors, host_tensors):
    """Time one whole request: upload, trace, download."""
    times = []
    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        _refill_device(host_tensors, device_tensors, cq_id=0)
        output = model.execute_trace(blocking=True)
        if output is not None:
            ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        times.append((time.perf_counter() - start) * 1000.0)
    return times


TIMERS = {
    "forward": _time_forward,
    "2cq": _time_2cq,
    "h2d_d2h": _time_h2d_d2h,
}


@pytest.mark.parametrize(
    "mesh_device",
    [(2, 1)] if detected_n300() else [1],
    indirect=True,
    ids=["n300_dp2"] if detected_n300() else ["single"],
)
@pytest.mark.parametrize(
    "device_params, mode",
    [
        (
            {
                "trace_region_size": 50_000_000,
                "num_command_queues": cqs,
                **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if detected_n300() else {}),
            },
            mode,
        )
        for mode, cqs in MODE_PARAMS
    ],
    indirect=["device_params"],
    ids=[mode for mode, _ in MODE_PARAMS],
)
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [(12, 8192)] if detected_n300() else [(1, 512), (8, 512), (16, 512), (32, 512)],
    ids=(lambda shapes: [f"b{b}_s{s}" for b, s in shapes])(
        [(12, 8192)] if detected_n300() else [(1, 512), (8, 512), (16, 512), (32, 512)]
    ),
)
@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "masked"])
def test_perf(mesh_device, batch_size, seq_len, masked, mode):
    """Report wall-clock time for the shape the local card runs.

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
        # The refilling modes need a host copy that carries the same shard layout.
        host_tensors = _stage_on_host(inputs, mesh_device) if mode != "forward" else {}

        out = model.forward(**device_tensors, no_padding=not masked)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(out)

        model.capture_trace(**device_tensors, mesh_device=mesh_device, cq_id=0, no_padding=not masked)
        for _ in range(3):
            model.execute_trace(blocking=True)

        times = TIMERS[mode](model, mesh_device, device_tensors, host_tensors)
        model.release_trace()

        times.sort()
        avg_ms = sum(times) / len(times)
        report_perf(batch_size, seq_len, valid_len, masked, times[0], avg_ms, mode=mode)
