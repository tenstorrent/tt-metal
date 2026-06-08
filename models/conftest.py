# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ast
import gc

import pytest
import torchvision.transforms as transforms
from PIL import Image


@pytest.fixture(autouse=True)
def ensure_gc():
    gc.collect()


import pytest
import torchvision.transforms as transforms
from PIL import Image


@pytest.fixture
def imagenet_label_dict():
    path = "models/sample_data/imagenet_class_labels.txt"
    with open(path, "r") as file:
        class_labels = ast.literal_eval(file.read())
    return class_labels


@pytest.fixture
def imagenet_sample_input():
    path = "models/sample_data/ILSVRC2012_val_00048736.JPEG"

    im = Image.open(path)
    im = im.resize((224, 224))
    return transforms.ToTensor()(im).unsqueeze(0)


@pytest.fixture
def mnist_sample_input():
    path = "models/sample_data/torchvision_mnist_digit_7.jpg"
    im = Image.open(path)
    return im


@pytest.fixture
def iam_ocr_sample_input():
    path = "models/sample_data/iam_ocr_image.jpg"
    im = Image.open(path)
    return im


@pytest.fixture
def hf_cat_image_sample_input():
    path = "models/sample_data/huggingface_cat_image.jpg"
    im = Image.open(path)
    return im


def pytest_sessionfinish(session, exitstatus):
    """Opt-in local perf/accuracy target validation (off by default).

    Enable with ``VALIDATE_PERF_TARGETS=1``. After the session completes — so the
    demos have written complete benchmark JSON — this runs the *same* validator
    used in CI (``.github/scripts/utils/validate_perf_targets.py``) against
    ``generated/benchmark_data`` and ``models/model_targets.yaml`` and fails the
    session if a target is missed. This is the local counterpart to the CI
    "Validate perf and accuracy targets" step; in-test ``verify_perf`` stays
    warning-only so it never aborts before data is written, and there is a single
    comparison implementation shared between local and CI.

    Knobs:
      VALIDATE_PERF_TARGETS=1                   enable
      VALIDATE_PERF_TARGETS_STRICT_MISSING=1    also fail on missing/TODO targets
      VALIDATE_PERF_TARGETS_SKU=<sku>           override SKU (else autodetect / JSON card_type)
    """
    import importlib.util
    import os
    from pathlib import Path

    from loguru import logger

    def _env_true(name: str) -> bool:
        return os.getenv(name, "0").strip().lower() not in {"", "0", "false", "no", "off"}

    if not _env_true("VALIDATE_PERF_TARGETS"):
        return

    repo_root = Path(__file__).resolve().parents[1]
    validator_path = repo_root / ".github/scripts/utils/validate_perf_targets.py"
    targets_yaml = repo_root / "models/model_targets.yaml"
    if not validator_path.exists() or not targets_yaml.exists():
        logger.warning(
            f"VALIDATE_PERF_TARGETS set but validator or targets file is missing "
            f"({validator_path}, {targets_yaml}); skipping perf validation."
        )
        return

    benchmark_dir = next(
        (
            candidate
            for candidate in (Path.cwd() / "generated/benchmark_data", repo_root / "generated/benchmark_data")
            if candidate.is_dir()
        ),
        None,
    )
    if benchmark_dir is None:
        logger.warning("VALIDATE_PERF_TARGETS set but no generated/benchmark_data directory found; skipping.")
        return

    # The .github dir is not an importable package, so load the validator core by path.
    spec = importlib.util.spec_from_file_location("validate_perf_targets", validator_path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)

    # SKU: explicit env > best-effort device autodetect > per-file card_type (None).
    sku_override = os.getenv("VALIDATE_PERF_TARGETS_SKU") or None
    if sku_override is None:
        try:
            from models.demos.utils.device_sku import get_current_device_sku_name

            sku_override = get_current_device_sku_name()
        except Exception as exc:  # device may be closed at session end; fall back to benchmark card_type
            logger.debug(f"Could not auto-detect SKU for perf validation ({exc}); relying on benchmark card_type.")

    result = validator.validate(
        targets_yaml_path=targets_yaml,
        benchmark_dir=benchmark_dir,
        tests_yaml_path=None,
        sku_override=sku_override,
    )

    strict_missing = _env_true("VALIDATE_PERF_TARGETS_STRICT_MISSING")

    for error in result.schema_errors:
        logger.error(f"[perf-validation] schema: {error}")
    for failure in result.hard_failures:
        logger.error(f"[perf-validation] {failure}")
    for missing in result.missing_entries:
        (logger.error if strict_missing else logger.warning)(f"[perf-validation] missing target: {missing}")

    failed = (
        bool(result.schema_errors) or bool(result.hard_failures) or (strict_missing and bool(result.missing_entries))
    )
    if failed:
        logger.error("[perf-validation] perf/accuracy target validation FAILED (VALIDATE_PERF_TARGETS=1).")
        if session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
    else:
        logger.info(
            f"[perf-validation] passed: {result.num_benchmark_files} benchmark file(s), "
            f"{len(result.hard_failures)} failures, {len(result.missing_entries)} missing/TODO."
        )
