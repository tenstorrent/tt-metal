# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Update 6's packaged boot class for the M7 batch profile (UPDATE6_PLAN §1.3; LEAD_DECISIONS_M7 M41, M292 O6, M328;
O6_RELEASE_ARCH_PROPOSAL.md 0002b639, critic da2a44b8): the served class v5.6 with update 5's packaged ring class
behind it, as update 5's package_boot_s3pb_ring_apc composes APC-v2 with it.

    Gemma4ResidentMultiM7Packaged(Gemma4ResidentMultiM7V56ForCausalLM, Gemma4ResidentMultiS32PBRingPackaged)

MRO: this class -> V56 -> V55 -> V53 (the release revision: l.251 M7_SERVING_CONFIG_SHA256 set) -> APC v3 -> APC v2 ->
APC v1 -> Gemma4ResidentMultiS32PBRingPackaged -> Ring -> S3-PB -> ... . From the packaged ring class it inherits, not
copies, initialize_vllm_model's ensure_resident_cache() (the engine's raw cache, keyed by ENGINE_PACKAGE) and
warmup_model_prefill's continuation pre-warm (contwarm, keyed by ENGINE_PACKAGE). This class adds:
  - ENGINE_PACKAGE = "resident_m5_14" (the ring class's "resident_m5_11" is update 5's engine);
  - the drafter cache (D3, M328): before the engine cache, ensure_drafter_cache() makes $GEMMA4_M7_DRAFTER_CACHE a
    complete resident_m7.drafter_weights.convert() layout (L*/, global/, vectors.pt, selector.pt) for the pinned DFlash2
    snapshot, built on the first start and checked on every start, with package_boot's marker, manifest, discard and
    .partial protocol (package_boot.py): the key is DFlash2's repo and revision (drafter_common.DFLASH2_ID /
    DFLASH2_REVISION, b0cd01b4), the sha256 of the converter modules (drafter_weights.py, drafter_common.py, common.py),
    the precision (bf8: quantize_layout's bfloat8_b) and the layer and entry counts.
It defines no serving method and no build_drafter: build_drafter resolves to V56's (M308; validator v9 V2).
"""

import fcntl
import hashlib
import os
import shutil
import time
from pathlib import Path

from loguru import logger

from models.demos.gemma4_31b_qb2.tt import package_boot as pb
from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_m7_v5_6_vllm import Gemma4ResidentMultiM7V56ForCausalLM
from models.demos.gemma4_31b_qb2.tt.package_boot_s3pb_ring import Gemma4ResidentMultiS32PBRingPackaged

DRAFTER_FORMAT = 1
DRAFTER_CONVERTER_MODULES = ("drafter_weights.py", "drafter_common.py", "common.py")
DRAFTER_PRECISION = "bf8"


def drafter_cache_key():
    """Everything a converted drafter cache depends on. Equal keys = interchangeable caches."""
    from models.demos.gemma4_31b_qb2 import resident_m7
    from models.demos.gemma4_31b_qb2.resident_m7 import drafter_common as dc

    pkg_dir = Path(resident_m7.__file__).parent
    h = hashlib.sha256()
    for name in DRAFTER_CONVERTER_MODULES:
        h.update(f"{name}\0{pb._sha256(pkg_dir / name)}\n".encode())
    return {"format": DRAFTER_FORMAT, "drafter_repo": dc.DFLASH2_ID, "drafter_revision": dc.DFLASH2_REVISION,
            "converter_sha256": h.hexdigest(), "precision": DRAFTER_PRECISION, "layers": dc.D_LAYERS,
            "entries": len(dc.D_ENTRIES)}


def ensure_drafter_cache(cache=None, verify=None, convert=None):
    """Make $GEMMA4_M7_DRAFTER_CACHE a complete drafter cache for the pinned DFlash2 snapshot; build it if not.

    Returns the cache path. `convert` replaces resident_m7.drafter_weights.convert (tests)."""
    cache = Path(cache or os.environ.get("GEMMA4_M7_DRAFTER_CACHE") or "")
    if not cache.name:
        raise ValueError("GEMMA4_M7_DRAFTER_CACHE must name the resident drafter cache directory")
    verify = verify or os.environ.get("GEMMA4_RESIDENT_CACHE_VERIFY", "sha256")
    if verify not in pb.VERIFY_MODES:
        raise ValueError(f"GEMMA4_RESIDENT_CACHE_VERIFY={verify!r}: expected one of {pb.VERIFY_MODES}")
    key = drafter_cache_key()
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache.with_name(cache.name + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        problem = pb.check_cache(cache, key, verify)
        if problem is None:
            logger.info("Resident drafter cache {}: complete for {}@{} ({})", cache, key["drafter_repo"],
                        key["drafter_revision"][:12], DRAFTER_PRECISION)
            return cache
        logger.info("Resident drafter cache {}: {}; converting {}@{} into it", cache, problem, key["drafter_repo"],
                    key["drafter_revision"][:12])
        pb.discard(cache)
        partial = cache.with_name(cache.name + ".partial")
        if partial.is_symlink():
            partial.unlink()
        elif partial.exists():
            shutil.rmtree(partial)
        if convert is None:
            from models.demos.gemma4_31b_qb2.resident_m7.drafter_weights import convert
        t0 = time.time()
        convert(partial)
        converted = time.time() - t0
        os.sync()  # every converted byte on disk before the marker that vouches for it
        files = pb.file_manifest(partial)
        pb._write_marker(partial, {"key": key, "files": files, "convert_seconds": round(converted, 1),
                                   "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        os.rename(partial, cache)
        pb._fsync_dir(cache.parent)
        logger.info("Resident drafter cache {}: converted in {:.0f} s ({} files, {:.2f} GB)", cache, converted,
                    len(files), sum(size for _, size, _ in files) / 1e9)
    return cache


class Gemma4ResidentMultiM7Packaged(Gemma4ResidentMultiM7V56ForCausalLM, Gemma4ResidentMultiS32PBRingPackaged):
    """Class v5.6 that first makes sure the drafter and engine caches exist, plus the continuation pre-warm (the
    package's batch class)."""

    ENGINE_PACKAGE = "resident_m5_14"

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, **kwargs):
        ensure_drafter_cache()
        return super().initialize_vllm_model(hf_config, mesh_device, max_batch_size, **kwargs)
