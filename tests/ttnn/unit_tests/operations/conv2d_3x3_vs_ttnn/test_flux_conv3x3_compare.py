# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""conv2d_3x3 vs ttnn.conv2d on the FLUX.2 VAE's 3x3 convs (models/tt_dit, 1024² and 2048² images).

Cases:
  - "1x1": the 31 distinct single-device convs of the decoder (33 calls per image) and encoder (25 calls).
  - "bh2x4": the decoder's per-device convs on the Blackhole (2, 4) preset: rows split 2 ways (mesh axis 0),
    channels split 4 ways (axis 1), the conv sees H/2 + 2 rows (neighbor_pad halo) with padding (0, 0, 1, 1).
  - "bh2x2": the same on the Blackhole (2, 2) preset (FLUX.2's QuietBox CI config): channels split 2 ways.
  - "bh4x8": the same on the Blackhole (4, 8) preset (FLUX.2's Galaxy CI config). Its axes are the other way
    round: channels split 4 ways (axis 0), rows split 8 ways (axis 1), so the conv sees H/8 + 2 rows.

ttnn.conv2d runs exactly as models/tt_dit/layers/conv2d.py runs it: bf16 TILE DRAM-interleaved NHWC input,
bf16 weights through prepare_conv_weights / prepare_conv_bias, HiFi4 + fp32 DEST, act_block_h_override 32,
config tensors in DRAM, and the model's DRAM width-slice table for the mesh shape.

conv2d_3x3 gets the same data, per-side padding and math config on its flattened [1, 1, H*W, C] contract.
FLUX_CMP_PROXY=1 runs its symmetric same-work proxy instead (tagged "proxy" in the results):
  - the encoder downsamplers (stride 2, padding (0, 1, 0, 1)) run padding 1, which has the same output size;
  - the bh2x4 halo convs ((H/2 + 2) x W, padding (0, 0, 1, 1)) run the H/2 interior rows with padding 1.
Each op's PCC is against its own torch reference, on bands of output rows (top, middle, bottom) so the
2048² references stay cheap. Failures of either op (unsupported, OOM, error) are recorded, not raised.

Per-call device time comes from tracy signposts `cmp|<case>|<op>|<i>`; run with `scripts/run_safe_pytest.sh
--profile --run-all` and select cases with FLUX_CMP_CASES (indices) or FLUX_CMP_GROUP (1x1 / bh2x4 / bh2x2 / bh4x8).
FLUX_CMP_SKIP_TTNN=1 / FLUX_CMP_SKIP_NEW=1 skip the ttnn.conv2d / conv2d_3x3 side. FLUX_CMP_LAYOUT=RM gives both ops
ROW_MAJOR input (the single-device convs after the decoder's upsamplers get it in the model).
"""

import itertools
import json
import os
import sys
import time
import traceback

import pytest
import torch
import ttnn
from tracy import signpost

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.insert(0, REPO)

from ttnn.operations.conv2d_3x3 import conv2d_3x3, prepare_conv2d_bias, prepare_conv2d_weights  # noqa: E402

from models.tt_dit.layers.conv2d import Conv2d as TtDitConv2d  # noqa: E402

FLUX_L1_SMALL_SIZE = 65536
MEASURED_ITERS = 3
RESULTS = os.environ.get("FLUX_CMP_RESULTS", "/tmp/flux_cmp_results.jsonl")
IMAGE_SIZES = (1024, 2048)
BAND_ROWS = 32
C_IN_GRANULE = 8


def _decoder_convs(image):
    """(name, H, W, C_in, C_out, stride, padding4, tensor_parallel) per call, in forward order."""
    size = image // 8
    convs = [("dec.conv_in", size, 32, 512, 1, True)]
    convs += [("dec.mid", size, 512, 512, 1, True)] * 4
    channels = [512, 512, 512, 256, 128]
    for i, (c_in, c_out) in enumerate(itertools.pairwise(channels)):
        for r in range(3):
            convs.append((f"dec.up{i}.res", size, c_in if r == 0 else c_out, c_out, 1, True))
            convs.append((f"dec.up{i}.res", size, c_out, c_out, 1, True))
        if i != len(channels) - 2:
            size *= 2
            convs.append((f"dec.up{i}.upsample", size, c_out, c_out, 1, True))
    convs.append(("dec.conv_out", size, 128, 3, 1, False))
    return [(n, s, s, ci, co, st, (1, 1, 1, 1), tp) for n, s, ci, co, st, tp in convs]


def _encoder_convs(image):
    size = image
    convs = [("enc.conv_in", size, 3, 128, 1, (1, 1, 1, 1))]
    channels = [128, 128, 256, 512, 512]
    for i, (c_in, c_out) in enumerate(itertools.pairwise(channels)):
        for r in range(2):
            convs.append((f"enc.down{i}.res", size, c_in if r == 0 else c_out, c_out, 1, (1, 1, 1, 1)))
            convs.append((f"enc.down{i}.res", size, c_out, c_out, 1, (1, 1, 1, 1)))
        if i != len(channels) - 2:
            convs.append((f"enc.down{i}.downsample", size, c_out, c_out, 2, (0, 1, 0, 1)))
            size //= 2
    convs += [("enc.mid", size, 512, 512, 1, (1, 1, 1, 1))] * 4
    convs.append(("enc.conv_out", size, 512, 32, 1, (1, 1, 1, 1)))
    return [(n, s, s, ci, co, st, p, True) for n, s, ci, co, st, p in convs]


def _per_device(conv, h_factor, tp_factor):
    """VaeConv2d's per-device conv: rows split with a 1-row halo each side, channels split on the larger side."""
    name, h, w, c_in, c_out, stride, (top, bottom, left, right), tp = conv
    assert stride == 1 and top == bottom == 1
    if tp:
        if c_out > c_in:
            c_out //= tp_factor
        else:
            c_in //= tp_factor
    return (name, h // h_factor + 2, w, c_in, c_out, stride, (0, 0, left, right), tp), (c_in, c_out)


def _build_cases():
    cases = {}

    def add(group, key, name, image, calls_key, model_channels):
        case = cases.setdefault(
            (group, key),
            {"group": group, "names": set(), "calls": {}, "model_channels": model_channels},
        )
        case["names"].add(name)
        case["calls"][calls_key] = case["calls"].get(calls_key, 0) + 1

    for image in IMAGE_SIZES:
        for part, convs in (("dec", _decoder_convs(image)), ("enc", _encoder_convs(image))):
            for conv in convs:
                name, h, w, c_in, c_out, stride, pad, _ = conv
                add("1x1", (h, w, c_in, c_out, stride, pad), name, image, f"{part}{image}", (c_in, c_out))
        for conv in _decoder_convs(image):
            dev, _ = _per_device(conv, h_factor=2, tp_factor=4)
            name, h, w, c_in, c_out, stride, pad, _ = dev
            add("bh2x4", (h, w, c_in, c_out, stride, pad), name, image, f"dec{image}", (conv[3], conv[4]))
        for conv in _decoder_convs(image):
            dev, _ = _per_device(conv, h_factor=2, tp_factor=2)
            name, h, w, c_in, c_out, stride, pad, _ = dev
            add("bh2x2", (h, w, c_in, c_out, stride, pad), name, image, f"dec{image}", (conv[3], conv[4]))
        for conv in _decoder_convs(image):
            dev, _ = _per_device(conv, h_factor=8, tp_factor=4)
            name, h, w, c_in, c_out, stride, pad, _ = dev
            add("bh4x8", (h, w, c_in, c_out, stride, pad), name, image, f"dec{image}", (conv[3], conv[4]))

    out = []
    for (group, (h, w, c_in, c_out, stride, pad)), case in cases.items():
        out.append(
            {
                "group": group,
                "H": h,
                "W": w,
                "c_in": c_in,
                "c_out": c_out,
                "stride": stride,
                "padding": pad,
                "names": sorted(case["names"]),
                "calls": case["calls"],
                "model_channels": case["model_channels"],
            }
        )
    groups = ("1x1", "bh2x4", "bh2x2", "bh4x8")
    out.sort(key=lambda c: (groups.index(c["group"]), c["H"] * c["W"] * c["c_in"] * c["c_out"]))
    for c in out:
        c["id"] = (
            f'{c["group"]}-{c["H"]}x{c["W"]}-{c["c_in"]}to{c["c_out"]}'
            + (f'-s{c["stride"]}' if c["stride"] != 1 else "")
            + ("" if c["padding"] == (1, 1, 1, 1) else "-p" + "".join(map(str, c["padding"])))
        )
    return out


CASES = _build_cases()
_ONLY = {int(i) for i in os.environ.get("FLUX_CMP_CASES", "").split(",") if i.strip()}
_GROUP = os.environ.get("FLUX_CMP_GROUP", "")


def _out_hw(h, w, stride, pad):
    top, bottom, left, right = pad
    return (h + top + bottom - 3) // stride + 1, (w + left + right - 3) // stride + 1


def _bands(h_out):
    if h_out <= 3 * BAND_ROWS:
        return [(0, h_out)]
    mid = h_out // 2 - BAND_ROWS // 2
    return [(0, BAND_ROWS), (mid, mid + BAND_ROWS), (h_out - BAND_ROWS, h_out)]


def _ref_rows(x_hwc, weight, bias, stride, pad, r0, r1):
    """Output rows [r0, r1) of the conv, from only the input rows they read."""
    top, bottom, left, right = pad
    h = x_hwc.shape[0]
    first = r0 * stride - top
    last = (r1 - 1) * stride - top + 3
    rows = x_hwc[max(first, 0) : min(last, h)].float().permute(2, 0, 1).unsqueeze(0)
    rows = torch.nn.functional.pad(rows, (left, right, max(0, -first), max(0, last - h)))
    return torch.nn.functional.conv2d(rows, weight.float(), bias.float(), stride=stride)[0].permute(1, 2, 0)


def _pcc_bands(out_flat, x_hwc, weight, bias, stride, pad, h_out, w_out, c_out):
    out = out_flat.reshape(-1, out_flat.shape[-1])[: h_out * w_out, :c_out].reshape(h_out, w_out, c_out)
    got, ref = [], []
    for r0, r1 in _bands(h_out):
        got.append(out[r0:r1].double().flatten())
        ref.append(_ref_rows(x_hwc, weight, bias, stride, pad, r0, r1).double().flatten())
    got, ref = torch.cat(got), torch.cat(ref)
    return torch.corrcoef(torch.stack([got, ref]))[0, 1].item()


def _ideal_us(h_out, w_out, c_in, c_out, cycles_per_tile_mac=64, cores=110, mhz=1350):
    """110-core HiFi4 ideal: 64 cycles per 32x32x32 tile-MAC (LoFi is 16)."""
    tile_macs = h_out * w_out * 9 * c_in * c_out / 32**3
    return tile_macs * cycles_per_tile_mac / cores / mhz


def _in_collect_pass():
    """True during run_safe_pytest's up-front collect pass, which records programs but runs nothing on device."""
    plugin = sys.modules.get("tests.plugins.up_front_collect")
    return plugin is not None and getattr(plugin, "_PASS", "real") == "collect"


def _measure(device, label, run):
    """One warm-up call, then MEASURED_ITERS signposted calls. Returns the last output.

    The closing `cmpend` signpost ends the last call's window, so ops issued after it (readback, the next
    case's warm-up) are not summed into it."""
    out = run()
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    for i in range(MEASURED_ITERS):
        ttnn.deallocate(out)
        signpost(header=f"cmp|{label}|{i}")
        out = run()
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
    signpost(header=f"cmpend|{label}")
    return out


def _status(e):
    msg = str(e)
    if "Out of Memory" in msg or "out of memory" in msg.lower():
        return "oom", msg.splitlines()[0][:300]
    if "not in SUPPORTED" in msg or "unsupported" in msg.lower():
        return "unsupported", msg.splitlines()[0][:300]
    return "error", (msg.splitlines()[0] if msg else type(e).__name__)[:300]


@pytest.mark.parametrize("device_params", [{"l1_small_size": FLUX_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("idx", range(len(CASES)), ids=[c["id"] for c in CASES])
def test_flux_conv3x3_compare(device, idx):
    case = CASES[idx]
    if _ONLY and idx not in _ONLY:
        pytest.skip("not selected by FLUX_CMP_CASES")
    if _GROUP and case["group"] != _GROUP:
        pytest.skip("not selected by FLUX_CMP_GROUP")
    H, W, c_in, c_out, stride, pad = (case[k] for k in ("H", "W", "c_in", "c_out", "stride", "padding"))
    h_out, w_out = _out_hw(H, W, stride, pad)
    record = {k: case[k] for k in ("id", "group", "H", "W", "c_in", "c_out", "stride", "names", "calls")}
    record["padding"] = list(pad)
    record["h_out"], record["w_out"] = h_out, w_out
    record["ideal_hifi4_us"] = _ideal_us(h_out, w_out, c_in, c_out)
    t0 = time.time()
    collect = _in_collect_pass()

    torch.manual_seed(0)
    # The collect pass only needs shapes; skipping the RNG saves seconds per 2048² case.
    x = torch.zeros(H, W, c_in, dtype=torch.bfloat16) if collect else torch.randn(H, W, c_in, dtype=torch.bfloat16)
    weight = (torch.randn(c_out, c_in, 3, 3) / (9 * c_in) ** 0.5).to(torch.bfloat16)
    bias = (0.1 * torch.randn(c_out)).to(torch.bfloat16)

    # --- ttnn.conv2d, as models/tt_dit/layers/conv2d.py calls it ---
    mesh_shape = {"1x1": (1, 1), "bh2x4": (2, 4), "bh2x2": (2, 2), "bh4x8": (4, 8)}[case["group"]]
    slice_table = TtDitConv2d.slice_params.get(mesh_shape, TtDitConv2d.slice_default)
    num_slices = slice_table.get((H, W, *case["model_channels"]))
    record["ttnn_num_slices"] = num_slices
    slice_config = (
        ttnn.Conv2dSliceConfig(num_slices=num_slices, slice_type=ttnn.Conv2dDRAMSliceWidth)
        if num_slices is not None
        else None
    )
    # On a mesh, VaeConv2d hands the conv the ROW_MAJOR output of its neighbor_pad halo exchange, so the per-device
    # (bh2x4) convs run on ROW_MAJOR input; FLUX_CMP_BH_LAYOUT=TILE measures them on TILE instead. On one device the
    # conv after each decoder upsampler gets the upsample's ROW_MAJOR output; FLUX_CMP_LAYOUT=RM measures that.
    bh_rm = case["group"] in ("bh2x4", "bh2x2", "bh4x8") and os.environ.get("FLUX_CMP_BH_LAYOUT", "RM") == "RM"
    rm_input = c_in < C_IN_GRANULE or bh_rm or os.environ.get("FLUX_CMP_LAYOUT") == "RM"
    record["input_layout"] = "ROW_MAJOR" if rm_input else "TILE"
    in_layout = ttnn.ROW_MAJOR_LAYOUT if rm_input else ttnn.TILE_LAYOUT
    tt_x = ttnn.from_torch(
        x.reshape(1, H, W, c_in),
        dtype=ttnn.bfloat16,
        layout=in_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    # act_block_h: the model pins 32. FLUX_CMP_TTNN_DEFAULT_ABH=1 measures a tuned baseline instead: ttnn's own
    # choice, and 32 only where ttnn's choice does not fit in L1.
    # Slices: the model's table is keyed without stride, so a stride-2 conv can get more slices than its output
    # width allows (TT_FATAL num_slices <= max_num_slices) and the model would crash there; such a case is
    # recorded and re-measured with ttnn's own slice choice.
    abh_options = [None, 32] if os.environ.get("FLUX_CMP_TTNN_DEFAULT_ABH") == "1" else [32]
    slice_options = [slice_config, None] if slice_config is not None else [None]
    failed_abh = set()
    ttnn_attempts = [(a, sl) for a in abh_options for sl in slice_options]
    if os.environ.get("FLUX_CMP_SKIP_TTNN") == "1":
        record["ttnn_status"] = "skipped"
        ttnn_attempts = []
    for abh, attempt_slice in ttnn_attempts:
        if abh in failed_abh:
            continue
        conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, config_tensors_in_dram=True)
        if abh is not None:
            conv_config.act_block_h_override = abh
        base = dict(
            in_channels=c_in,
            out_channels=c_out,
            batch_size=1,
            input_height=H,
            input_width=W,
            kernel_size=(3, 3),
            stride=(stride, stride),
            padding=pad,
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=conv_config,
        )
        try:
            tt_w = ttnn.prepare_conv_weights(
                weight_tensor=ttnn.from_torch(weight, dtype=ttnn.bfloat16),
                weights_format="OIHW",
                input_memory_config=tt_x.memory_config(),
                input_layout=tt_x.get_layout(),
                has_bias=True,
                input_dtype=tt_x.dtype,
                slice_config=attempt_slice,
                **base,
            )
            tt_w = ttnn.to_device(tt_w, device)
            tt_b = ttnn.prepare_conv_bias(
                bias_tensor=ttnn.from_torch(bias.reshape(1, 1, 1, c_out), dtype=ttnn.bfloat16),
                input_memory_config=tt_x.memory_config(),
                input_layout=tt_x.get_layout(),
                input_dtype=tt_x.dtype,
                **base,
            )
            tt_b = ttnn.to_device(tt_b, device)

            def run_ref():
                out, _, _ = ttnn.conv2d(
                    input_tensor=tt_x,
                    weight_tensor=tt_w,
                    bias_tensor=tt_b,
                    compute_config=compute_config,
                    slice_config=attempt_slice,
                    return_output_dim=True,
                    return_weights_and_bias=True,
                    **{k: v for k, v in base.items() if k != "groups"},
                )
                return out

            ref_out = _measure(device, f'{case["id"]}|ttnn.conv2d', run_ref)
            record["ttnn_status"] = "ok"
            record.pop("ttnn_error", None)
            record["ttnn_act_block_h_override"] = abh or 0
            if attempt_slice is None and slice_config is not None:
                record["ttnn_num_slices"] = "auto"
            if not collect:
                record["pcc_ttnn"] = _pcc_bands(
                    ttnn.to_torch(ref_out), x, weight, bias, stride, pad, h_out, w_out, c_out
                )
            ttnn.deallocate(ref_out)
            ttnn.deallocate(tt_w)
            ttnn.deallocate(tt_b)
            break
        except Exception as e:  # recorded, the case still measures conv2d_3x3
            record["ttnn_status"], record["ttnn_error"] = _status(e)
            traceback.print_exc()
            msg = str(e)
            if "max_num_slices" in msg and attempt_slice is not None:
                record["ttnn_model_table_error"] = record["ttnn_error"]
            elif abh is None and any(t in msg for t in ("beyond max L1", "clash with L1 buffers", "Out of Memory")):
                record["ttnn_default_abh_error"] = record["ttnn_error"]
                failed_abh.add(abh)
            else:
                break
    ttnn.deallocate(tt_x)

    # --- conv2d_3x3 on the same data (FLUX_CMP_PROXY=1: the symmetric same-work proxy) ---
    if os.environ.get("FLUX_CMP_SKIP_NEW") == "1":
        record["new_status"] = "skipped"
        if not collect:
            with open(RESULTS, "a") as f:
                f.write(json.dumps(record) + "\n")
        return
    new_pad = pad
    new_x = x
    if pad != (1, 1, 1, 1) and os.environ.get("FLUX_CMP_PROXY") == "1":
        record["new_proxy"] = True
        new_pad = (1, 1, 1, 1)
        if stride == 1:
            new_x = x[1:-1]
    new_h = new_x.shape[0]
    c_in_new = -(-c_in // C_IN_GRANULE) * C_IN_GRANULE
    if c_in_new != c_in:
        new_x = torch.nn.functional.pad(new_x, (0, c_in_new - c_in))
    new_weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, c_in_new - c_in))
    try:
        new_in = ttnn.from_torch(
            new_x.reshape(1, 1, new_h * W, c_in_new),
            dtype=ttnn.bfloat16,
            layout=in_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        prep = dict(
            stride=stride,
            padding=new_pad,
            dilation=1,
            input_dtype=ttnn.bfloat16,
            input_layout=in_layout,
            weight_dtype=ttnn.bfloat16,
            input_memory_config=new_in.memory_config(),
            device=device,
        )
        new_w = prepare_conv2d_weights(new_weight, **prep)
        new_b = prepare_conv2d_bias(bias, **prep)
        new_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )

        def run_new():
            return conv2d_3x3(
                new_in,
                new_w,
                input_height=new_h,
                input_width=W,
                bias=new_b,
                padding=new_pad,
                stride=stride,
                compute_kernel_config=new_cfg,
            )

        new_out = _measure(device, f'{case["id"]}|conv2d_3x3', run_new)
        record["new_status"] = "ok"
        nh, nw = _out_hw(new_h, W, stride, new_pad)
        if not collect:
            record["pcc_new"] = _pcc_bands(
                ttnn.to_torch(new_out), new_x, new_weight, bias, stride, new_pad, nh, nw, c_out
            )
        ttnn.deallocate(new_out)
        ttnn.deallocate(new_in)
    except Exception as e:  # recorded: an unsupported or OOM case is a result here
        record["new_status"], record["new_error"] = _status(e)
        traceback.print_exc()

    record["wall_s"] = round(time.time() - t0, 1)
    if collect:
        return
    with open(RESULTS, "a") as f:
        f.write(json.dumps(record) + "\n")
    print("FLUX_CMP:", json.dumps(record))
