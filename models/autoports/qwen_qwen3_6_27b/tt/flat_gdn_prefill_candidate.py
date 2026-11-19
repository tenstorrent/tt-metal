# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Coherent opt-in compact GDN packed producer/consumer, no default wiring.

Importing this module uses only the Python standard library. Device calls are
lazy and reserved for the parent's serialized hardware test. No host readback,
runtime output cache, policy mutation, or state publication between chunks.
"""

from .prefill_conv_candidate import prefill_sources


def flat_prefill_conv(qkv, history, taps, *, batch, seq, cores=None, poison_dead_padding=False):
    """Read compact [1,B*S,2560] rows; emit only temporal Q/K/V padded by user.

    Kernel address arithmetic and compute are the batched prefill candidate's;
    only the input logical-shape check changes. No whole packed reshape.
    """
    import ttnn
    from .optimized_decoder import (
        _conv107__CONV_RM_PIPE2_READER,
        _conv107__CONV_RM_PIPE2_COMPUTE,
        _conv107__CONV_RM_PIPE2_WRITER,
    )

    channels = 2560
    assert tuple(qkv.shape) == (1, batch * seq, channels)
    assert 1 <= batch <= 32 and seq > 0 and channels == 2560
    assert batch * seq * 3 < 2**32
    assert tuple(history.shape) == (batch, 3, channels)
    assert qkv.layout == ttnn.TILE_LAYOUT and history.layout == ttnn.ROW_MAJOR_LAYOUT
    assert len(taps) == 4
    raw = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)
    inputs = [raw, history, *taps]
    for value in inputs:
        assert value.layout == ttnn.ROW_MAJOR_LAYOUT and value.dtype == ttnn.bfloat16
        assert value.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert all(tuple(t.shape) == (1, 1, channels) for t in taps)

    def allocate(shape):
        return ttnn.empty(
            shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=qkv.device(),
        )

    result = allocate((batch, seq, channels))
    carry = allocate((batch, 3, channels))
    grid = qkv.device().compute_with_storage_grid_size()
    tasks = batch * seq * 3
    ncores = min(tasks, cores or grid.x * grid.y)
    assert 1 <= ncores <= grid.x * grid.y
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    start = 0
    for i, coord in enumerate(coords):
        count = tasks // ncores + (i < tasks % ncores)
        rr[coord.x][coord.y] = [t.buffer_address() for t in inputs] + [start, count, seq]
        wr[coord.x][coord.y] = [carry.buffer_address(), result.buffer_address(), start, count, seq]
        cr[coord.x][coord.y] = [count]
        start += count
    rct, wct = [], []
    for value in inputs:
        rct.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    for value in (carry, result):
        wct.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())

    def kernel(source, config, args, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=args,
            config=config,
        )

    reader, compute, writer = prefill_sources(
        _conv107__CONV_RM_PIPE2_READER,
        _conv107__CONV_RM_PIPE2_COMPUTE,
        _conv107__CONV_RM_PIPE2_WRITER,
        poison_dead_padding=poison_dead_padding,
    )
    kernels = [
        kernel(reader, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(writer, ttnn.WriterConfigDescriptor(), wr, wct),
        kernel(
            compute,
            ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
            ),
            cr,
            [],
        ),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in (*range(8), 9, 10)
    ]
    ttnn.generic_op(inputs + [result, carry], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    # Dispatch ordering supplies the cross-core barrier; never alias input and
    # output history inside the kernel. Stable state allocation remains intact.
    ttnn.copy(carry, history)
    result = ttnn.to_layout(result, ttnn.TILE_LAYOUT)
    padded = ((seq + 31) // 32) * 32
    if padded != seq:
        result = ttnn.pad(result, ((0, 0), (0, padded - seq), (0, 0)), value=0.0)
    return result[:, :, :512], result[:, :, 512:1024], result[:, :, 1024:2560]


def flat_packed_gdn(layer, packed, state, *, batch, seq, cores=None):
    """Consume [1,B*S,4160]; preserve compact Z and output projection rows.

    Q/K/V and the small beta/g scalars alone acquire per-user padded sequence
    axes. The existing FP32 chunk recurrence and normalization are unchanged.
    The scheduler still owns logical chunking, list/resume, and BF16 publication.
    """
    import ttnn

    assert layer.kind == "linear_attention" and batch > 1 and seq > 1
    assert tuple(packed.shape) == (1, batch * seq, 4160)
    assert packed.dtype == ttnn.bfloat16 and packed.layout == ttnn.TILE_LAYOUT
    assert tuple(state.conv.shape) == (batch, 3, 2560)
    assert tuple(state.recurrent.shape) == (batch, 12, 128, 128)
    assert state.recurrent.dtype == ttnn.float32, "Caller must retain live FP32 state across prefill chunks"
    if layer.policy.get("tiled_prefill_conv", False):
        from .tiled_prefill_conv_candidate import tiled_prefill_conv

        q, k, v = tiled_prefill_conv(
            packed,
            state.conv,
            layer.taps,
            batch=batch,
            seq=seq,
            cores=cores,
            block_ct=layer.policy.get("prefill_conv_block_ct", 8),
        )
    else:
        q, k, v = flat_prefill_conv(packed[:, :, :2560], state.conv, layer.taps_rm, batch=batch, seq=seq, cores=cores)
    beta = ttnn.reshape(ttnn.sigmoid(packed[:, :, 4096:4108]), (batch, seq, 12))
    g = ttnn.reshape(layer._decay(packed[:, :, 4128:4140]), (batch, seq, 12))
    padded = ((seq + 31) // 32) * 32
    if seq != padded:
        beta, g = (ttnn.pad(t, ((0, 0), (0, padded - seq), (0, 0)), value=0.0) for t in (beta, g))
    out, updated = ttnn.transformer.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state.recurrent,
        output_final_state=True,
        chunk_size=32,
        num_key_heads=4,
        output_head_major=True,
        eye=layer.eye,
        tril=layer.lower,
        ones=layer.ones,
        masks=layer.masks,
        compute_kernel_config=layer.ckc,
    )
    # FP32 to FP32 copy matches the existing live chunk state. Publication to
    # the externally owned BF16 state remains in flat_prefill_forward.
    ttnn.typecast(updated, state.recurrent.dtype, output_tensor=state.recurrent)
    out = ttnn.reshape(out, ttnn.Shape((batch, 12, seq, 128)), ttnn.Shape((batch, 12, padded, 128)))
    out = ttnn.rms_norm(
        ttnn.typecast(out, ttnn.bfloat16),
        weight=layer.weights["linear_attn.norm.weight"],
        epsilon=layer.config.rms_norm_eps,
        compute_kernel_config=layer.ckc,
    )
    # Gather heads once, drop per-user tile padding before token-independent
    # gating, and keep the out projection and following residual compact.
    out = ttnn.reshape(ttnn.transformer.concatenate_heads(out), (1, batch * seq, 1536))
    out = ttnn.multiply(ttnn.silu(packed[:, :, 2560:4096]), out)
    return layer._linear(out, "linear_attn.out_proj.weight")


def install(layer, *, cores=None):
    """Install only after the parent adds the documented flat_chunk hook."""
    from .flat_prefill import _flat_chunk

    assert layer.kind == "linear_attention" and layer.policy.get("flat_prefill")
    assert "_flat_packed_gdn" in _flat_chunk.__code__.co_names, "Add the documented optional producer/consumer hook"
    assert not hasattr(layer, "_flat_packed_gdn")
    layer._flat_packed_gdn = lambda packed, state, *, batch, seq: flat_packed_gdn(
        layer, packed, state, batch=batch, seq=seq, cores=cores
    )
    layer.policy["flat_packed_gdn_candidate"] = True
    return layer


def cpu_check():
    """Address/contract checks only, with no torch or TTNN imports."""
    import ast
    import json
    import sys
    from pathlib import Path
    from .prefill_conv_candidate import cpu_check as check_conv

    check_conv()
    own = ast.parse(Path(__file__).read_text())
    original = ast.parse(Path(__file__).with_name("prefill_conv_candidate.py").read_text())
    # Duplicate launcher differs only in public shape contract. Prove it has
    # not silently changed kernel program/config/CB semantics relative to the
    # independently tested regular batching candidate.
    f = next(n for n in own.body if isinstance(n, ast.FunctionDef) and n.name == "flat_prefill_conv")
    g = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "batched_prefill_conv")

    # Everything after the shape checks is source-identical AST.
    def tail(node):
        i = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assert) and "1 <= batch" in ast.unparse(n))
        return [ast.dump(n, include_attributes=False) for n in node.body[i:]]

    assert tail(f) == tail(g)
    checks = 0
    for batch in (2, 3, 17, 32):
        for seq in (2, 17, 31, 32, 33, 65, 257, 513):
            padded = ((seq + 31) // 32) * 32
            for user in range(batch):
                for time in range(seq):
                    flat_row = user * seq + time
                    assert divmod(flat_row, seq) == (user, time)
                    padded_row = user * padded + time
                    assert divmod(padded_row, padded) == (user, time)
            # Corrections and Z remain inside their original exact packed slots;
            # beta and g ignore the original BA padding slots [4108:4128/4140:4160].
            spans = ((0, 512), (512, 1024), (1024, 2560), (2560, 4096), (4096, 4108), (4128, 4140))
            assert all(0 <= a < b <= 4160 for a, b in spans)
            assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
            checks += 1
    assert "torch" not in sys.modules and "ttnn" not in sys.modules
    print(json.dumps({"status": "pass", "scope": "CPU source/address/layout contract only", "cases": checks}))


def full_layer(argv):
    """Run the existing TP4 layer harness after installing this instance hook."""
    import argparse
    import hashlib
    import json
    import sys
    import traceback
    from pathlib import Path
    from unittest.mock import patch

    parser = argparse.ArgumentParser(description=full_layer.__doc__)
    parser.add_argument("--cores", type=int)
    parser.add_argument("--output", "--candidate-output", dest="candidate_output", type=Path, required=True)
    args, forwarded = parser.parse_known_args(argv)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    root = Path(__file__).resolve().parents[1]
    assert args.candidate_output.resolve().is_relative_to((root / "doc/optimized_multichip_decoder").resolve())
    assert not args.candidate_output.exists()
    assert "--baseline" not in forwarded and "--output" not in forwarded
    assert args.cores is None or args.cores >= 1
    from models.autoports.qwen_qwen3_6_27b.tests import run_multichip_decoder as runner

    original = runner.MultichipDecoder.from_state_dict
    counters = {"instances": 0, "calls": 0}

    def create(cls, *positional, **keywords):
        model = install(original(*positional, **keywords), cores=args.cores)
        selected = model._flat_packed_gdn

        def invoke(*a, **kw):
            counters["calls"] += 1
            return selected(*a, **kw)

        model._flat_packed_gdn = invoke
        counters["instances"] += 1
        return model

    failure = None
    try:
        with (
            patch.object(runner.MultichipDecoder, "from_state_dict", classmethod(create)),
            patch.object(sys, "argv", [sys.argv[0], *forwarded, "--output", str(args.candidate_output)]),
        ):
            runner.main()
        assert counters["instances"] and counters["calls"], "Choose B>1, S>1 with flat_prefill enabled"
    except BaseException:
        failure = traceback.format_exc()
        raise
    finally:
        paths = [
            Path(__file__),
            root / "tt/prefill_conv_candidate.py",
            root / "tt/recurrent_next_candidate.py",
            root / "tt/flat_prefill.py",
            root / "tt/multichip_decoder.py",
            root / "tt/optimized_decoder.py",
        ]
        evidence = {
            "kind": "flat_packed_gdn_instance_hook",
            **counters,
            "cores": args.cores,
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "failure": failure,
            "arguments": forwarded,
        }
        report = json.loads(args.candidate_output.read_text()) if args.candidate_output.exists() else {"status": "fail"}
        report["flat_packed_gdn_candidate"] = evidence
        args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--cpu-check"]:
        cpu_check()
    else:
        full_layer(sys.argv[1:])
