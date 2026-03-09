#!/usr/bin/env python3
"""
Standalone repro for teardown hang in unet decoder_2 block (ops 30-35).

Bisect found: adding decoder_2 (ops 30-35) to
encoder_1+encoder_3+encoder_4+bottleneck+decoder_1 causes TEARDOWN_HANG.

Uses real conv2d ops to produce correctly-formatted BLOCK_SHARDED inputs
for conv_transpose2d and the skip connection, matching the actual unet pipeline.

Decoder_2 ops:
  Op 30: conv_transpose2d (512->256, 32x32->64x64, HEIGHT_SHARDED output)
  Op 31: to_memory_config (reshard conv2d_7 skip to HEIGHT_SHARDED)
  Op 32: concat (conv_transpose2d + skip, HEIGHT_SHARDED)
  Op 33: to_memory_config (reshard to BLOCK_SHARDED for conv2d)
  Op 34: conv2d (512->256, 64x64, BLOCK_SHARDED output)
  Op 35: conv2d (256->256, 64x64, HEIGHT_SHARDED output)
"""

import os
import sys

tt_metal_root = os.environ.get(
    "TT_METAL_RUNTIME_ROOT", os.path.join(os.environ.get("TT_MLIR_HOME", ""), "third_party/tt-metal/src/tt-metal")
)
ttnn_path = os.path.join(tt_metal_root, "ttnn")
if ttnn_path not in sys.path:
    sys.path.insert(0, ttnn_path)

import torch
import ttnn


def open_device():
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape((1, 1)),
        l1_small_size=1 << 15,
    )
    return device


def close_device(device):
    print("Closing device...", flush=True)
    ttnn.close_mesh_device(device)
    print("Device closed.", flush=True)


# ---- Memory configs ----

DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)

# Op 29 output (conv2d_17) = input to op 30: BLOCK_SHARDED [128, 64] on 8x8
CONV17_OUT_MEM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
        [128, 64],
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)

# Op 13 output (conv2d_7) = skip connection: BLOCK_SHARDED [512, 32] on 8x8
CONV7_OUT_MEM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
        [512, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)

# Op 30 output / Op 31 output: HEIGHT_SHARDED [64, 256] on 8x8
HEIGHT_SHARDED_64_256 = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
        [64, 256],
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)

# Op 32 output (concat): HEIGHT_SHARDED [64, 512] on 8x8
CONCAT_OUT_MEM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
        [64, 512],
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)

# Op 33 output: BLOCK_SHARDED [608, 96] on 6x7
BLOCK_SHARDED_608_96 = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 6))]),
        [608, 96],
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)

# Op 34 output: BLOCK_SHARDED [512, 32] on 8x8
CONV18_OUT_MEM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
        [512, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)

# Op 35 output: HEIGHT_SHARDED [64, 256] on 8x8
CONV19_OUT_MEM = HEIGHT_SHARDED_64_256

# ---- Conv configs (from unet codegen) ----

CONV_TRANSPOSE_CONFIG = ttnn.Conv2dConfig(
    weights_dtype=ttnn.DataType.BFLOAT16,
    deallocate_activation=True,
    config_tensors_in_dram=True,
    act_block_h_override=64,
    enable_kernel_stride_folding=False,
)

CONV18_CONFIG = ttnn.Conv2dConfig(
    weights_dtype=ttnn.DataType.BFLOAT16,
    activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
    deallocate_activation=True,
    config_tensors_in_dram=True,
    act_block_h_override=32,
    enable_kernel_stride_folding=False,
)

CONV19_CONFIG = ttnn.Conv2dConfig(
    weights_dtype=ttnn.DataType.BFLOAT16,
    activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
    deallocate_activation=True,
    config_tensors_in_dram=True,
    act_block_h_override=64,
    enable_kernel_stride_folding=False,
)


def make_dram_tensor(shape, device):
    """Create a random bfloat16 tensor in DRAM INTERLEAVED on device."""
    t = torch.randn(shape, dtype=torch.bfloat16)
    host = ttnn.from_torch(t)
    host = host.to(ttnn.Layout.TILE)
    return host.to(device, DRAM)


def prepare_all_weights(device):
    """Prepare weights for producer conv2ds and decoder_2 ops."""
    weights = {}

    # (Op 30 conv_transpose2d skipped — simulated via to_memory_config)

    # --- Op 34: conv2d_18: 512->256, kernel [3,3] ---
    print("Preparing conv2d_18 weights (op 34)...", flush=True)
    w = torch.randn([256, 512, 3, 3], dtype=torch.bfloat16)
    w_host = ttnn.from_torch(w).to(ttnn.Layout.ROW_MAJOR)
    weights["w34"] = ttnn.prepare_conv_weights(
        weight_tensor=w_host,
        input_memory_config=BLOCK_SHARDED_608_96,
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=256,
        batch_size=1,
        input_height=64,
        input_width=64,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=CONV18_CONFIG,
        compute_config=None,
        slice_config=None,
    )
    b = torch.randn([1, 1, 1, 256], dtype=torch.bfloat16)
    b_host = ttnn.from_torch(b).to(ttnn.Layout.ROW_MAJOR)
    weights["b34"] = ttnn.prepare_conv_bias(
        bias_tensor=b_host,
        input_memory_config=BLOCK_SHARDED_608_96,
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=256,
        batch_size=1,
        input_height=64,
        input_width=64,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=CONV18_CONFIG,
        compute_config=None,
    )

    # --- Op 35: conv2d_19: 256->256, kernel [3,3] ---
    print("Preparing conv2d_19 weights (op 35)...", flush=True)
    w = torch.randn([256, 256, 3, 3], dtype=torch.bfloat16)
    w_host = ttnn.from_torch(w).to(ttnn.Layout.ROW_MAJOR)
    weights["w35"] = ttnn.prepare_conv_weights(
        weight_tensor=w_host,
        input_memory_config=CONV18_OUT_MEM,
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=64,
        input_width=64,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=CONV19_CONFIG,
        compute_config=None,
        slice_config=None,
    )
    b = torch.randn([1, 1, 1, 256], dtype=torch.bfloat16)
    b_host = ttnn.from_torch(b).to(ttnn.Layout.ROW_MAJOR)
    weights["b35"] = ttnn.prepare_conv_bias(
        bias_tensor=b_host,
        input_memory_config=CONV18_OUT_MEM,
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=64,
        input_width=64,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=CONV19_CONFIG,
        compute_config=None,
    )

    print("All weights prepared.", flush=True)
    return weights


def produce_inputs(device, weights):
    """Create input tensors matching the real unet memory layouts."""
    # conv2d_17: [1, 1, 1024, 512] in BLOCK_SHARDED [128, 64] on 8x8
    # (conv_transpose2d requires BLOCK_SHARDED input to pick the right execution path)
    print("Creating conv2d_17 input (DRAM -> BLOCK_SHARDED)...", flush=True)
    conv2d_17_dram = make_dram_tensor([1, 1, 1024, 512], device)
    conv2d_17 = ttnn.to_memory_config(conv2d_17_dram, CONV17_OUT_MEM)
    ttnn.deallocate(conv2d_17_dram, False)
    print(f"  conv2d_17 shape: {conv2d_17.shape}", flush=True)

    # conv2d_7 skip connection: [1, 1, 4096, 256] in DRAM
    # Op 31 (to_memory_config) will reshard to HEIGHT_SHARDED
    print("Creating conv2d_7 skip in DRAM...", flush=True)
    conv2d_7 = make_dram_tensor([1, 1, 4096, 256], device)
    print(f"  conv2d_7 shape: {conv2d_7.shape}", flush=True)

    return conv2d_17, conv2d_7


def run_decoder2(device, weights, conv2d_17, conv2d_7):
    """Run decoder_2 ops (30-35).

    Op 30 (conv_transpose2d) can't be run standalone due to internal tensor format
    requirements. We simulate its output: [1, 1, 4096, 256] HEIGHT_SHARDED [64, 256] 8x8.
    """
    # Op 30: simulate conv_transpose2d output
    # Real op: 512->256, 32x32->64x64, output [1, 1, 4096, 256] HEIGHT_SHARDED
    # We create the equivalent by resharding the DRAM tensor
    print("[30] simulated conv_transpose2d output (DRAM -> HEIGHT_SHARDED)...", flush=True)
    ttnn.deallocate(conv2d_17, False)  # don't need the original input
    conv_transpose2d_1_dram = make_dram_tensor([1, 1, 4096, 256], device)
    conv_transpose2d_1 = ttnn.to_memory_config(conv_transpose2d_1_dram, HEIGHT_SHARDED_64_256)
    ttnn.deallocate(conv_transpose2d_1_dram, False)
    print(f"  -> shape {conv_transpose2d_1.shape}", flush=True)

    # Op 31: to_memory_config (reshard skip)
    print("[31] to_memory_config (reshard skip to HEIGHT_SHARDED)...", flush=True)
    to_mem_2 = ttnn.to_memory_config(conv2d_7, HEIGHT_SHARDED_64_256)
    ttnn.deallocate(conv2d_7, False)
    print(f"  -> shape {to_mem_2.shape}", flush=True)

    # Op 32: concat
    print("[32] concat...", flush=True)
    concat_1 = ttnn.concat([conv_transpose2d_1, to_mem_2], 3, memory_config=CONCAT_OUT_MEM)
    ttnn.deallocate(to_mem_2, False)
    ttnn.deallocate(conv_transpose2d_1, False)
    print(f"  -> shape {concat_1.shape}", flush=True)

    # Op 33: to_memory_config
    print("[33] to_memory_config (reshard to BLOCK_SHARDED)...", flush=True)
    to_mem_3 = ttnn.to_memory_config(concat_1, BLOCK_SHARDED_608_96)
    ttnn.deallocate(concat_1, False)
    print(f"  -> shape {to_mem_3.shape}", flush=True)

    # Op 34: conv2d_18
    print("[34] conv2d (512->256, 64x64)...", flush=True)
    conv2d_18 = ttnn.conv2d(
        input_tensor=to_mem_3,
        weight_tensor=weights["w34"],
        device=device,
        in_channels=512,
        out_channels=256,
        batch_size=1,
        input_height=64,
        input_width=64,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=weights["b34"],
        conv_config=CONV18_CONFIG,
        compute_config=None,
        slice_config=None,
        memory_config=CONV18_OUT_MEM,
    )
    print(f"  -> shape {conv2d_18.shape}", flush=True)

    # Op 35: conv2d_19
    print("[35] conv2d (256->256, 64x64)...", flush=True)
    conv2d_19 = ttnn.conv2d(
        input_tensor=conv2d_18,
        weight_tensor=weights["w35"],
        device=device,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=64,
        input_width=64,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=weights["b35"],
        conv_config=CONV19_CONFIG,
        compute_config=None,
        slice_config=None,
        memory_config=CONV19_OUT_MEM,
    )
    print(f"  -> shape {conv2d_19.shape}", flush=True)

    ttnn.deallocate(conv2d_19, False)
    print("All decoder_2 ops done.", flush=True)


def run_test(device, weights, label, op_list):
    """Run a subset of decoder_2 ops and test teardown.

    op_list: which ops to run. e.g. [30,31] or [30,31,32,33,34,35]
    Ops not in the list are skipped, intermediates are deallocated.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"TEST: {label} (ops {op_list})", flush=True)
    print(f"{'='*60}", flush=True)

    # Create fresh inputs
    conv_transpose2d_1_dram = make_dram_tensor([1, 1, 4096, 256], device)
    conv2d_7 = make_dram_tensor([1, 1, 4096, 256], device)

    conv_transpose2d_1 = None
    to_mem_2 = None
    concat_1 = None
    to_mem_3 = None
    conv2d_18 = None
    conv2d_19 = None

    # Op 30: simulate conv_transpose2d → HEIGHT_SHARDED
    if 30 in op_list:
        print("[30] to_memory_config (simulate conv_transpose2d)...", flush=True)
        conv_transpose2d_1 = ttnn.to_memory_config(conv_transpose2d_1_dram, HEIGHT_SHARDED_64_256)
        ttnn.deallocate(conv_transpose2d_1_dram, False)
    else:
        ttnn.deallocate(conv_transpose2d_1_dram, False)

    # Op 31: reshard skip → HEIGHT_SHARDED
    if 31 in op_list:
        print("[31] to_memory_config (skip → HEIGHT_SHARDED)...", flush=True)
        to_mem_2 = ttnn.to_memory_config(conv2d_7, HEIGHT_SHARDED_64_256)
        ttnn.deallocate(conv2d_7, False)
    else:
        ttnn.deallocate(conv2d_7, False)

    # Op 32: concat
    if 32 in op_list and conv_transpose2d_1 is not None and to_mem_2 is not None:
        print("[32] concat...", flush=True)
        concat_1 = ttnn.concat([conv_transpose2d_1, to_mem_2], 3, memory_config=CONCAT_OUT_MEM)
        ttnn.deallocate(to_mem_2, False)
        ttnn.deallocate(conv_transpose2d_1, False)
    else:
        if conv_transpose2d_1 is not None:
            ttnn.deallocate(conv_transpose2d_1, False)
        if to_mem_2 is not None:
            ttnn.deallocate(to_mem_2, False)

    # Op 33: reshard → BLOCK_SHARDED
    if 33 in op_list and concat_1 is not None:
        print("[33] to_memory_config (→ BLOCK_SHARDED)...", flush=True)
        to_mem_3 = ttnn.to_memory_config(concat_1, BLOCK_SHARDED_608_96)
        ttnn.deallocate(concat_1, False)
    elif concat_1 is not None:
        ttnn.deallocate(concat_1, False)

    # Op 34: conv2d 512->256
    if 34 in op_list and to_mem_3 is not None:
        print("[34] conv2d (512->256)...", flush=True)
        conv2d_18 = ttnn.conv2d(
            input_tensor=to_mem_3,
            weight_tensor=weights["w34"],
            device=device,
            in_channels=512,
            out_channels=256,
            batch_size=1,
            input_height=64,
            input_width=64,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=weights["b34"],
            conv_config=CONV18_CONFIG,
            compute_config=None,
            slice_config=None,
            memory_config=CONV18_OUT_MEM,
        )
    elif to_mem_3 is not None:
        ttnn.deallocate(to_mem_3, False)

    # Op 35: conv2d 256->256
    if 35 in op_list and conv2d_18 is not None:
        print("[35] conv2d (256->256)...", flush=True)
        conv2d_19 = ttnn.conv2d(
            input_tensor=conv2d_18,
            weight_tensor=weights["w35"],
            device=device,
            in_channels=256,
            out_channels=256,
            batch_size=1,
            input_height=64,
            input_width=64,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=weights["b35"],
            conv_config=CONV19_CONFIG,
            compute_config=None,
            slice_config=None,
            memory_config=CONV19_OUT_MEM,
        )
    elif conv2d_18 is not None:
        ttnn.deallocate(conv2d_18, False)

    if conv2d_19 is not None:
        ttnn.deallocate(conv2d_19, False)

    print(f"Ops done. Closing device...", flush=True)


def run_op35_isolation(device, weights):
    """Isolate op 35 — each test gets a fresh device to avoid state contamination."""
    import subprocess, time

    tests = {
        "test_a": "Full pipeline 30-35, HEIGHT_SHARDED output (original hang)",
        "test_b": "Full pipeline 30-35, DRAM output on op 35",
        "test_c": "Op 35 alone, DRAM input, HEIGHT_SHARDED output",
        "test_d": "Ops 30-34 only (no op 35)",
    }

    if hasattr(run_op35_isolation, "_subtest"):
        # We're running a specific sub-test (called from subprocess)
        subtest = run_op35_isolation._subtest

        if subtest == "test_a":
            # Full pipeline, HEIGHT_SHARDED output
            _run_pipeline(device, weights, use_dram_output=False)
        elif subtest == "test_b":
            # Full pipeline, DRAM output on op 35
            _run_pipeline(device, weights, use_dram_output=True)
        elif subtest == "test_c":
            # Op 35 alone with HEIGHT_SHARDED output
            _run_conv35_alone(device, weights, use_dram_output=False)
        elif subtest == "test_d":
            # Ops 30-34 only
            run_test(device, weights, "ops 30-34", [30, 31, 32, 33, 34])
        return

    # Run each test as a separate subprocess with fresh device
    script = os.path.abspath(__file__)
    for test_id, desc in tests.items():
        print(f"\n{'='*60}", flush=True)
        print(f"{test_id}: {desc}", flush=True)
        print(f"{'='*60}", flush=True)

        # Reset device
        subprocess.run(["tt-smi", "-r", "0"], capture_output=True, timeout=30)
        time.sleep(3)

        env = os.environ.copy()
        proc = subprocess.Popen(
            [sys.executable, "-u", script, "--test", f"_sub_{test_id}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=120)
            print(stdout, end="", flush=True)
            if proc.returncode == 0:
                print(f"  >>> RESULT: CLEAN EXIT", flush=True)
            else:
                print(f"  >>> RESULT: CRASHED (rc={proc.returncode})", flush=True)
                if "TT_FATAL" in stderr:
                    for line in stderr.splitlines():
                        if "TT_FATAL" in line:
                            print(f"  {line.strip()[:120]}", flush=True)
                            break
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"  >>> RESULT: HANG (timeout 120s)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("All isolation tests complete.", flush=True)
    print(f"{'='*60}", flush=True)
    # Skip normal close since we didn't run ops on this device
    sys.exit(0)


def _run_pipeline(device, weights, use_dram_output):
    """Full ops 30-35 pipeline, optionally with DRAM output on op 35."""
    ct1_dram = make_dram_tensor([1, 1, 4096, 256], device)
    c7 = make_dram_tensor([1, 1, 4096, 256], device)
    print("[30] to_memory_config...", flush=True)
    ct1 = ttnn.to_memory_config(ct1_dram, HEIGHT_SHARDED_64_256)
    ttnn.deallocate(ct1_dram, False)
    print("[31] to_memory_config...", flush=True)
    tm2 = ttnn.to_memory_config(c7, HEIGHT_SHARDED_64_256)
    ttnn.deallocate(c7, False)
    print("[32] concat...", flush=True)
    cat = ttnn.concat([ct1, tm2], 3, memory_config=CONCAT_OUT_MEM)
    ttnn.deallocate(tm2, False)
    ttnn.deallocate(ct1, False)
    print("[33] to_memory_config...", flush=True)
    tm3 = ttnn.to_memory_config(cat, BLOCK_SHARDED_608_96)
    ttnn.deallocate(cat, False)
    print("[34] conv2d (512->256)...", flush=True)
    c18 = ttnn.conv2d(
        input_tensor=tm3,
        weight_tensor=weights["w34"],
        device=device,
        in_channels=512,
        out_channels=256,
        batch_size=1,
        input_height=64,
        input_width=64,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=weights["b34"],
        conv_config=CONV18_CONFIG,
        compute_config=None,
        slice_config=None,
        memory_config=CONV18_OUT_MEM,
    )
    out_mem = DRAM if use_dram_output else CONV19_OUT_MEM
    out_label = "DRAM" if use_dram_output else "HEIGHT_SHARDED"
    print(f"[35] conv2d (256->256) → {out_label}...", flush=True)
    c19 = ttnn.conv2d(
        input_tensor=c18,
        weight_tensor=weights["w35"],
        device=device,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=64,
        input_width=64,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=weights["b35"],
        conv_config=CONV19_CONFIG,
        compute_config=None,
        slice_config=None,
        memory_config=out_mem,
    )
    print(f"  shape: {c19.shape}", flush=True)
    ttnn.deallocate(c19, False)
    print("Pipeline done.", flush=True)


def _run_conv35_alone(device, weights, use_dram_output):
    """Run only conv2d 256->256 with DRAM input."""
    w = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(torch.randn([256, 256, 3, 3], dtype=torch.bfloat16)).to(ttnn.Layout.ROW_MAJOR),
        input_memory_config=DRAM,
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=64,
        input_width=64,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=CONV19_CONFIG,
        compute_config=None,
        slice_config=None,
    )
    b = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(torch.randn([1, 1, 1, 256], dtype=torch.bfloat16)).to(ttnn.Layout.ROW_MAJOR),
        input_memory_config=DRAM,
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=64,
        input_width=64,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=CONV19_CONFIG,
        compute_config=None,
    )
    out_mem = DRAM if use_dram_output else CONV19_OUT_MEM
    inp = make_dram_tensor([1, 1, 4096, 256], device)
    out_label = "DRAM" if use_dram_output else "HEIGHT_SHARDED"
    print(f"conv2d 256->256, DRAM input → {out_label}...", flush=True)
    out = ttnn.conv2d(
        input_tensor=inp,
        weight_tensor=w,
        device=device,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=64,
        input_width=64,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=b,
        conv_config=CONV19_CONFIG,
        compute_config=None,
        slice_config=None,
        memory_config=out_mem,
    )
    print(f"  shape: {out.shape}", flush=True)
    ttnn.deallocate(out, False)
    print("Done.", flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        help="Which test: 'all' (ops 30-35), 'bisect' (progressive), " "or comma-separated ops like '30,31'",
    )
    args = parser.parse_args()

    # Handle subprocess sub-tests (each gets fresh device)
    if args.test.startswith("_sub_"):
        subtest = args.test[5:]  # strip "_sub_"
        device = open_device()
        device.enable_program_cache()
        weights = prepare_all_weights(device)
        run_op35_isolation._subtest = subtest
        run_op35_isolation(device, weights)
        device.disable_and_clear_program_cache()
        close_device(device)
        print("DONE - clean exit!", flush=True)
        return

    device = open_device()
    device.enable_program_cache()
    weights = prepare_all_weights(device)

    if args.test == "bisect":
        # Progressive tests — each creates fresh tensors, same device
        tests = [
            ("ops 30+31 only (two HEIGHT_SHARDED to_memory_config)", [30, 31]),
            ("ops 30+31+32 (+ concat)", [30, 31, 32]),
            ("ops 30+31+32+33 (+ reshard to BLOCK)", [30, 31, 32, 33]),
            ("ops 30+31+32+33+34 (+ conv2d 512->256)", [30, 31, 32, 33, 34]),
            ("ops 30-35 (full decoder_2)", [30, 31, 32, 33, 34, 35]),
        ]
        for label, ops in tests:
            run_test(device, weights, label, ops)
    elif args.test == "test_a":
        # Full pipeline 30-35, HEIGHT_SHARDED output (original hang)
        _run_pipeline(device, weights, use_dram_output=False)
    elif args.test == "test_b":
        # Full pipeline 30-35, DRAM output on op 35
        _run_pipeline(device, weights, use_dram_output=True)
    elif args.test == "test_c":
        # Op 35 alone, DRAM input, HEIGHT_SHARDED output
        _run_conv35_alone(device, weights, use_dram_output=False)
    elif args.test == "test_d":
        # Ops 30-34 only (no op 35)
        run_test(device, weights, "ops 30-34", [30, 31, 32, 33, 34])
    elif args.test == "test_concat_dram":
        # Concat with DRAM output instead of HEIGHT_SHARDED
        print("Ops 30+31 (to_memory_config) + concat with DRAM output", flush=True)
        a = make_dram_tensor([1, 1, 4096, 256], device)
        b = make_dram_tensor([1, 1, 4096, 256], device)
        print("concat with DRAM output...", flush=True)
        cat = ttnn.concat([a, b], 3, memory_config=DRAM)
        ttnn.deallocate(b, False)
        ttnn.deallocate(a, False)
        print(f"  shape: {cat.shape}", flush=True)
        ttnn.deallocate(cat, False)
        print("Done.", flush=True)
    elif args.test == "test_concat_sharded":
        # Concat with HEIGHT_SHARDED output (minimal repro)
        print("Concat two HEIGHT_SHARDED [64,256] → HEIGHT_SHARDED [64,512]", flush=True)
        a = make_dram_tensor([1, 1, 4096, 256], device)
        b = make_dram_tensor([1, 1, 4096, 256], device)
        a_s = ttnn.to_memory_config(a, HEIGHT_SHARDED_64_256)
        ttnn.deallocate(a, False)
        b_s = ttnn.to_memory_config(b, HEIGHT_SHARDED_64_256)
        ttnn.deallocate(b, False)
        print("concat with HEIGHT_SHARDED output...", flush=True)
        cat = ttnn.concat([a_s, b_s], 3, memory_config=CONCAT_OUT_MEM)
        ttnn.deallocate(b_s, False)
        ttnn.deallocate(a_s, False)
        print(f"  shape: {cat.shape}", flush=True)
        ttnn.deallocate(cat, False)
        print("Done.", flush=True)
    elif args.test == "all":
        run_test(device, weights, "full decoder_2", [30, 31, 32, 33, 34, 35])
    else:
        ops = [int(x) for x in args.test.split(",")]
        run_test(device, weights, f"ops {ops}", ops)

    device.disable_and_clear_program_cache()
    print("\nClosing device...", flush=True)
    close_device(device)
    print("DONE - clean exit!", flush=True)


if __name__ == "__main__":
    main()
