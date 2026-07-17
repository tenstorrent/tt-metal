#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


HARNESS_SOURCE = r'''\
# tracer_test_harness.py
# Stable runtime harness for running op-level PCC checks from a tracer manifest.
#
# Design goals:
# - Minimal generator output (no giant formatted strings).
# - No brace-escaping issues: generator does not embed large templates.
# - Per-record pytest parametrization supported via record_id.
#
# This harness assumes the activation layout contract used by the tracer artifacts:
# - Inputs/outputs stored as torch tensors in NCHW.
# - Runtime conversion to TTNN "activation" uses: NCHW -> [N, 1, H*W, C]
#
# Artifact path contract (see tools/bringup/README.md):
# - All artifact paths in the manifest are resolved relative to the directory
#   containing the manifest file. Absolute paths are used as-is.
#
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import torch
import ttnn


@dataclass(frozen=True)
class Record:
    idx: int
    kind: str
    in_shape: List[int]
    out_shape: List[int]
    params: Dict[str, Any]
    in_path: str
    out_path: str
    w_path: Optional[str] = None
    b_path: Optional[str] = None


def load_manifest(manifest_path: Path) -> List[Record]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    recs: List[Record] = []
    for i, r in enumerate(data.get("records", [])):
        recs.append(
            Record(
                idx=i,
                kind=str(r.get("kind")),
                in_shape=list(r.get("in_shape", [])),
                out_shape=list(r.get("out_shape", [])),
                params=dict(r.get("params", {}) or {}),
                in_path=str(r.get("in_path")),
                out_path=str(r.get("out_path")),
                w_path=r.get("w_path"),
                b_path=r.get("b_path"),
            )
        )
    return recs


def _resolve_artifact_path(manifest_path: Path, artifact_path: str) -> Path:
    # Option 1: artifact paths are manifest-relative; absolute paths used as-is.
    p = Path(artifact_path)
    if p.is_absolute():
        return p
    return Path(manifest_path).parent / p


def _load_torch_tensor(path: Path) -> torch.Tensor:
    t = torch.load(path, map_location="cpu")
    if isinstance(t, torch.Tensor):
        return t
    if isinstance(t, dict):
        for v in t.values():
            if isinstance(v, torch.Tensor):
                return v
    raise TypeError(f"Unsupported tensor payload type at {path}: {type(t)}")


def _nchw_to_hw_c(x: torch.Tensor) -> torch.Tensor:
    # N, C, H, W -> N, 1, H*W, C
    if x.dim() != 4:
        raise ValueError(f"Expected 4D NCHW tensor, got shape={tuple(x.shape)}")
    n, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).contiguous().view(n, 1, h * w, c)


def _hw_c_to_nchw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # N, 1, H*W, C -> N, C, H, W
    if x.dim() != 4:
        raise ValueError(f"Expected 4D [N,1,HW,C] tensor, got shape={tuple(x.shape)}")
    n, one, hw, c = x.shape
    if one != 1:
        raise ValueError(f"Expected dim1==1, got {one} for shape={tuple(x.shape)}")
    if hw != h * w:
        raise ValueError(f"Expected HW={h*w}, got {hw} for shape={tuple(x.shape)}")
    return x.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()


def _center_crop_to_shape(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    # Crop center of NCHW tensor to target H/W (no padding).
    if x.dim() != 4:
        raise ValueError(f"Expected NCHW, got shape={tuple(x.shape)}")
    _, _, h, w = x.shape
    if target_h > h or target_w > w:
        raise ValueError(f"Cannot crop from {(h, w)} to {(target_h, target_w)}")
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return x[:, :, top:top + target_h, left:left + target_w]


def _crop_to_match(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Strictly NCHW crop to the smaller spatial size; channels must already match.
    if a.dim() != 4 or b.dim() != 4:
        raise ValueError("crop_to_match expects NCHW tensors")
    if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1]:
        raise ValueError(f"Batch/channels mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    ah, aw = int(a.shape[2]), int(a.shape[3])
    bh, bw = int(b.shape[2]), int(b.shape[3])
    th, tw = min(ah, bh), min(aw, bw)
    return _center_crop_to_shape(a, th, tw), _center_crop_to_shape(b, th, tw)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    # Pearson correlation coefficient. Robust to constant tensors.
    a1 = a.detach().to(dtype=torch.float64).flatten()
    b1 = b.detach().to(dtype=torch.float64).flatten()
    if a1.numel() != b1.numel():
        raise ValueError(f"PCC requires same numel, got {a1.numel()} vs {b1.numel()}")

    a1 = a1 - a1.mean()
    b1 = b1 - b1.mean()
    denom = torch.linalg.vector_norm(a1) * torch.linalg.vector_norm(b1)
    if float(denom) == 0.0:
        return 1.0 if torch.allclose(a, b, rtol=0, atol=0) else 0.0
    return float((a1 @ b1) / denom)


def _parse_device_id(env_name: str) -> Optional[int]:
    raw = os.environ.get(env_name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as e:
        raise RuntimeError(f"{env_name} must be an int, got: {raw!r}") from e


def open_device(
    *,
    device_env: str = "TT_DEVICE_ID_OP_TEST",
    legacy_device_env: str = "TT_DEVICE_ID",
    default_device_id: int = 0,
    device_candidates: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7),
    verbose: bool = True,
) -> Tuple[Any, int]:
    """
    Open a Tenstorrent device deterministically:
    - Prefer device_env
    - Fall back to legacy_device_env
    - Else use default_device_id
    - If open fails, try device_candidates (in order), excluding the chosen id
    """
    chosen = _parse_device_id(device_env)
    source = device_env
    if chosen is None:
        chosen = _parse_device_id(legacy_device_env)
        source = legacy_device_env
    if chosen is None:
        chosen = int(default_device_id)
        source = "default"

    tried: List[int] = []
    last_exc: Optional[BaseException] = None

    def _try(dev_id: int):
        nonlocal last_exc
        tried.append(dev_id)
        if verbose:
            print(f"[ttnn] Trying open_device(device_id={dev_id})")
        try:
            dev = ttnn.open_device(device_id=dev_id)
            if verbose:
                print(f"[ttnn] Opened device_id={dev_id} (source={source}, env={os.environ.get(device_env)!r})")
            return dev
        except BaseException as e:
            last_exc = e
            if verbose:
                print(f"[ttnn] Failed to open device_id={dev_id}: {e}")
            return None

    dev = _try(chosen)
    if dev is not None:
        return dev, chosen

    for cand in device_candidates:
        cand = int(cand)
        if cand == chosen or cand in tried:
            continue
        dev = _try(cand)
        if dev is not None:
            return dev, cand

    raise RuntimeError(
        f"Failed to open any TT device. Tried={tried}. "
        f"Check env {device_env} / {legacy_device_env} or device locks."
    ) from last_exc


def close_device(dev: Any) -> None:
    try:
        ttnn.close_device(dev)
    except Exception:
        pass


def run_record(
    *,
    manifest_path: Path,
    record: Record,
    tt_device: Any,
    pcc_threshold: float,
    allow_crop: bool = False,
) -> float:
    kind = record.kind
    in_shape = record.in_shape
    out_shape = record.out_shape
    params = record.params

    if len(in_shape) != 4:
        raise ValueError(f"Expected 4D in_shape for rec_id={record.idx}, got {in_shape}")
    if len(out_shape) != 4:
        raise ValueError(f"Expected 4D out_shape for rec_id={record.idx}, got {out_shape}")

    n, c, h, w = map(int, in_shape)
    on, oc, oh, ow = map(int, out_shape)

    # Load artifacts
    x = _load_torch_tensor(_resolve_artifact_path(manifest_path, record.in_path))
    y_ref = _load_torch_tensor(_resolve_artifact_path(manifest_path, record.out_path))

    # Shape sanity checks (strict)
    if tuple(x.shape) != (n, c, h, w):
        raise ValueError(
            f"Input tensor shape mismatch for rec_id={record.idx}: expected {(n, c, h, w)} got {tuple(x.shape)}"
        )
    if tuple(y_ref.shape) != (on, oc, oh, ow):
        raise ValueError(
            f"Reference tensor shape mismatch for rec_id={record.idx}: expected {(on, oc, oh, ow)} got {tuple(y_ref.shape)}"
        )

    if kind == "Conv2d":
        w_t = _load_torch_tensor(_resolve_artifact_path(manifest_path, record.w_path)) if record.w_path else None
        b_t = _load_torch_tensor(_resolve_artifact_path(manifest_path, record.b_path)) if record.b_path else None

        x_act = _nchw_to_hw_c(x)
        x_tt = ttnn.from_torch(x_act, device=tt_device, layout=ttnn.ROW_MAJOR_LAYOUT)

        y_tt = ttnn.conv2d(
            input_tensor=x_tt,
            weight_tensor=w_t,
            bias_tensor=b_t,
            in_channels=int(params["in_channels"]),
            out_channels=int(params["out_channels"]),
            kernel_size=tuple(params["kernel_size"]),
            stride=tuple(params["stride"]),
            padding=tuple(params["padding"]),
            dilation=tuple(params["dilation"]),
            groups=int(params["groups"]),
        )
        y = ttnn.to_torch(y_tt)

        # IMPORTANT: reshape back using OUT shape, not input shape
        y_nchw = _hw_c_to_nchw(y, h=oh, w=ow)

    elif kind == "ReLU":
        x_act = _nchw_to_hw_c(x)
        x_tt = ttnn.from_torch(x_act, device=tt_device, layout=ttnn.TILE_LAYOUT)

        y_tt = ttnn.relu(x_tt)
        y = ttnn.to_torch(y_tt)

        y_nchw = _hw_c_to_nchw(y, h=h, w=w)

    else:
        pytest.skip(f"Unsupported op kind in harness: {kind}")

    # Strict shape match by default
    if y_nchw.shape != y_ref.shape:
        if allow_crop:
            y_nchw, y_ref2 = _crop_to_match(y_nchw, y_ref)
        else:
            raise AssertionError(
                f"Output shape mismatch for rec_id={record.idx} kind={kind}: "
                f"got {tuple(y_nchw.shape)} vs ref {tuple(y_ref.shape)}. "
                f"Consider allow_crop=True only if you intentionally permit cropping."
            )
    else:
        y_ref2 = y_ref

    score = pcc(y_nchw, y_ref2)
    if score < float(pcc_threshold):
        raise AssertionError(
            f"PCC={score:.6f} < {float(pcc_threshold):.6f} "
            f"(rec_id={record.idx}, kind={kind}, in_shape={in_shape}, out_shape={out_shape})"
        )
    return score


def run_record_id(
    *,
    manifest_path: Path,
    rec_id: int,
    tt_device: Any,
    pcc_threshold: float,
    allow_crop: bool = False,
) -> float:
    recs = load_manifest(manifest_path)
    if rec_id < 0 or rec_id >= len(recs):
        raise IndexError(f"rec_id out of range: {rec_id} (num_records={len(recs)})")
    return run_record(
        manifest_path=manifest_path,
        record=recs[int(rec_id)],
        tt_device=tt_device,
        pcc_threshold=pcc_threshold,
        allow_crop=allow_crop,
    )

'''


def _parse_only(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    raw = value.replace(",", " ").split()
    return [x.strip() for x in raw if x.strip()]


def _load_manifest_records(manifest_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return list(data.get("records", []))


def _select_record_ids(records: Sequence[Dict[str, Any]], only: Optional[List[str]]) -> List[int]:
    if not only:
        return list(range(len(records)))
    only_set = set(only)
    ids: List[int] = []
    for i, r in enumerate(records):
        kind = str(r.get("kind"))
        if kind in only_set:
            ids.append(i)
    return ids


TEST_FILE_TEMPLATE = """# Auto-generated minimal op-level PCC pytest
# Generated by: phase2_generate_tests_simplified.py
#
# How to run:
#   # ensure tracer_test_harness.py is in the same folder as this test
#   pytest -q {out_name}

from __future__ import annotations

from pathlib import Path

import pytest

from tracer_test_harness import open_device, close_device, run_record_id


MANIFEST_PATH = Path(r"{manifest_path}")
RECORD_IDS = {record_ids}
ONLY = {only}
PCC_THRESHOLD = {pcc_threshold}
ALLOW_CROP = {allow_crop}

DEVICE_ENV = {device_env!r}
LEGACY_DEVICE_ENV = "TT_DEVICE_ID"
DEFAULT_DEVICE_ID = {default_device_id}
DEVICE_CANDIDATES = {device_candidates}


@pytest.fixture(scope="session")
def tt_device():
    dev, dev_id = open_device(
        device_env=DEVICE_ENV,
        legacy_device_env=LEGACY_DEVICE_ENV,
        default_device_id=DEFAULT_DEVICE_ID,
        device_candidates=DEVICE_CANDIDATES,
        verbose=True,
    )
    print(f"[ttnn] Using device_id={{dev_id}}")
    try:
        yield dev
    finally:
        close_device(dev)


@pytest.mark.parametrize("rec_id", RECORD_IDS, ids=[f"rec{{i}}" for i in RECORD_IDS])
def test_op_pcc(tt_device, rec_id: int):
    run_record_id(
        manifest_path=MANIFEST_PATH,
        rec_id=int(rec_id),
        tt_device=tt_device,
        pcc_threshold=float(PCC_THRESHOLD),
        allow_crop=bool(ALLOW_CROP),
    )
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate minimal op-level PCC pytest from tracer manifest.")
    ap.add_argument("--manifest", required=True, help="Path to tracer manifest.json")
    ap.add_argument("--out-test", required=True, help="Output pytest file path")
    ap.add_argument("--only", default=None, help="Comma/space-separated op kinds to include (e.g. Conv2d,ReLU)")
    ap.add_argument("--pcc-threshold", type=float, default=0.999, help="PCC threshold (default: 0.999)")
    ap.add_argument(
        "--allow-crop", action="store_true", help="Allow center-cropping when output shapes mismatch (NOT recommended)"
    )
    ap.add_argument("--device-env", default="TT_DEVICE_ID_OP_TEST", help="Env var used to select TT device")
    ap.add_argument("--default-device-id", type=int, default=0, help="Default TT device id if env vars not set")
    ap.add_argument("--device-candidates", default="0,1,2,3,4,5,6,7", help="Comma-separated fallback device IDs")
    ap.add_argument("--write-harness", action="store_true", help="Write tracer_test_harness.py next to the out-test")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    out_test = Path(args.out_test).resolve()
    out_test.parent.mkdir(parents=True, exist_ok=True)

    only = _parse_only(args.only)
    records = _load_manifest_records(manifest_path)
    record_ids = _select_record_ids(records, only)

    cand = [int(x.strip()) for x in str(args.device_candidates).split(",") if x.strip()]

    content = TEST_FILE_TEMPLATE.format(
        out_name=out_test.name,
        manifest_path=str(manifest_path),
        record_ids=record_ids,
        only=only,
        pcc_threshold=float(args.pcc_threshold),
        allow_crop=bool(args.allow_crop),
        device_env=str(args.device_env),
        default_device_id=int(args.default_device_id),
        device_candidates=cand,
    )

    out_test.write_text(content, encoding="utf-8")
    print(f"Wrote pytest: {out_test}")
    print(f"Records selected: {len(record_ids)} / {len(records)} (only={only})")

    if args.write_harness:
        harness_path = out_test.parent / "tracer_test_harness.py"
        harness_path.write_text(HARNESS_SOURCE, encoding="utf-8")
        print(f"Wrote harness: {harness_path}")


if __name__ == "__main__":
    main()
