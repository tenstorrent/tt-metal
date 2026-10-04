"""Pure profile/budget checks and pass-through observation of stock serving construction."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat

GIB = 1024**3
MODEL = "Llama-3.1-8B-Instruct"
SNAPSHOT = "0e9e39f249a16976918f6564b8830bc894c89659"
BASELINE = "21754c005c015161fe5de2d1cdb10bf64a6fd5a1"
CANDIDATE = "ee072dd216c2d19c6b228be26f695757d8ec1500"


def budget(checkpoint_bytes=16071906850):
    # Original T3K1x8 serving: 128256x4096 embedding, 32 layers,
    # KV (capacity131072 +32 output reservations*64) / block64 =>2080 blocks, one local KV head, head dim128.
    embedding = 128256 * 4096 * 2
    kv = 32 * 2 * 8 * 2080 * 1 * 64 * 128 * 2
    cache_estimate = checkpoint_bytes + 7 * embedding + kv + 8 * GIB
    cache_cap = 2 ** math.ceil(math.log2(math.ceil(cache_estimate * 1.25)))
    memory_estimate = 2 * checkpoint_bytes + 2 * 8 * embedding + 8 * GIB
    memory_cap = 2 ** math.ceil(math.log2(memory_estimate))
    assert cache_cap == memory_cap == 64 * GIB
    return {
        "checkpoint_logical_byte_bound": checkpoint_bytes,
        "embedding_bytes": embedding,
        "kv_bf16_eight_replica_upper_bytes": kv,
        "cache_estimate_bytes": cache_estimate,
        "cache_cap_bytes": cache_cap,
        "memory_estimate_bytes": memory_estimate,
        "memory_limit_bytes": memory_cap,
        "runtime_cache_allowance_bytes": 8 * GIB,
        "encoding_margin_fraction": 0.25,
        "live_memory_floor_bytes": 8 * GIB,
        "disk_reserve_bytes": 34 * GIB,
        "source_parameters": {
            "vocab": 128256,
            "dim": 4096,
            "layers": 32,
            "mesh": [1, 8],
            "blocks": 2080,
            "token_capacity": 131072,
            "output_reservation_blocks": 32,
            "local_kv_heads": 1,
            "block_size": 64,
            "head_dim": 128,
        },
    }


def available_memory(path=Path("/proc/meminfo")):
    rows = dict(line.split(":", 1) for line in path.read_text().splitlines())
    value, units = rows["MemAvailable"].split()
    assert units == "kB"
    return int(value) * 1024


def regular_bytes(root):
    assert root.is_dir() and not root.is_symlink()
    total = 0
    for path in root.rglob("*"):
        info = path.lstat()
        assert stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode), "Cache link/special file refused"
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


def tensor_bound(tensor):
    dtype = str(tensor.dtype).lower().split(".")[-1]
    width = {"bfloat16": 2, "bfloat8_b": 2, "bfloat4_b": 2, "float32": 4, "int32": 4, "uint32": 4, "uint16": 2}[dtype]
    dims = list(tensor.padded_shape)
    assert 0 < len(dims) <= 8 and all(isinstance(d, int) and d > 0 for d in dims)
    # BF4/8 packed blocks occupy less than BF16; 8 replicas, padded physical
    # shape, 25% encoding margin and1MiB metadata. Refuse unsupported dtypes.
    return math.ceil(math.prod(dims) * width * 8 * 1.25) + 1024**2


def cap_dump(original, root, receipt, memory=available_memory):
    """Serialize only the declared producer's owned cache, retaining exact dump return."""
    import fcntl

    root, receipt = Path(root), Path(receipt)
    limits = budget()

    def bounded(file_name, tensor, *args, **kwargs):
        path = Path(file_name)
        assert path.is_absolute() and path.parent.resolve().is_relative_to(root.resolve())
        assert not path.is_symlink()
        # One cold server owns the entire cache; flock also serializes any threads.
        with (root.parent / ".producer-cap.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            used, bound = regular_bytes(root), tensor_bound(tensor)
            free = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
            assert used + bound <= limits["cache_cap_bytes"], "Declared producer cache byte cap"
            assert free >= bound + limits["disk_reserve_bytes"], "Producer live disk reserve"
            assert memory() >= limits["live_memory_floor_bytes"], "Producer live host memory reserve"
            result = original(file_name, tensor, *args, **kwargs)
            actual = regular_bytes(root)
            assert (
                actual <= limits["cache_cap_bytes"] and path.is_file() and path.stat().st_size <= bound
            ), "Serialization exceeded source-supported bound"
            with receipt.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "path": str(path.relative_to(root)),
                            "file_bytes": path.stat().st_size,
                            "upper_bytes": bound,
                            "cache_bytes": actual,
                            "pid": os.getpid(),
                        }
                    )
                    + "\n"
                )
            return result

    return bounded


def profile(result, groups):
    models, arguments = result
    assert len(models) == len(arguments) == 1
    args = arguments[0]
    dtypes = [
        str(args.optimizations.get_tensor_dtype(i, group, prefetcher=bool(args.prefetcher)))
        for i in range(args.n_layers)
        for group in groups
    ]
    variant = args._weight_cache_build_variant()
    assert variant["precision"] == hashlib.sha1("|".join(dtypes).encode()).hexdigest()[:12]
    row = {
        "kind": "stock-vllm-profile-v1",
        "model_name": args.model_name,
        "n_layers": args.n_layers,
        "model_dimensions": {
            name: getattr(args, name) for name in ("dim", "vocab_size", "n_heads", "n_kv_heads", "head_dim")
        },
        "mesh_shape": list(args.mesh_device.shape),
        "instruct": args.instruct,
        "max_seq_len": args.max_seq_len,
        "dtype": str(models[0].dtype),
        "build_variant": variant,
        "tensor_dtypes": dtypes,
        "dummy_weights": args.dummy_weights,
    }
    validate_profile(row)
    return row


def validate_profile(row):
    assert row["kind"] == "stock-vllm-profile-v1" and row["model_name"] == MODEL and row["n_layers"] == 32
    assert (
        row["mesh_shape"] == [1, 8]
        and row["max_seq_len"] == 32768
        and row["instruct"] is True
        and row["dummy_weights"] is False
    )
    assert row["dtype"].lower().split(".")[-1] == "bfloat8_b"
    assert row["model_dimensions"] == {
        "dim": 4096,
        "vocab_size": 128256,
        "n_heads": 32,
        "n_kv_heads": 8,
        "head_dim": 128,
    }
    variant = row["build_variant"]
    assert set(variant) == {"prefetcher", "precision", "batch", "fused_ag", "hf_rope"}
    assert variant["batch"] == 32 and variant["prefetcher"] is False and variant["hf_rope"] is False
    assert isinstance(variant["fused_ag"], bool) and len(variant["precision"]) == 12
    assert (
        len(row["tensor_dtypes"]) == 32 * 6
        and variant["precision"] == hashlib.sha1("|".join(row["tensor_dtypes"]).encode()).hexdigest()[:12]
    )
    return row


def observe(original, target, groups, source, phase):
    assert source in (BASELINE, CANDIDATE) and phase in ("producer", "baseline", "candidate")

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        row = profile(result, groups)
        with Path(target).open("x") as stream:
            stream.write(
                json.dumps(
                    {"profile": row, "source": source, "phase": phase, "pid": os.getpid(), "return_passthrough": True},
                    indent=2,
                )
                + "\n"
            )
        return result

    return observed


def install_observer(target, source, phase):
    """Observe at the stock import time; do not force an early model import."""
    import importlib.abc
    import importlib.machinery
    import sys

    fullname = "models.tt_transformers.tt.generator_vllm"

    def attach(module):
        assert not hasattr(module, "_formatter_profile_observed"), "Duplicate profile observer"
        module.initialize_vllm_text_transformer = observe(
            module.initialize_vllm_text_transformer, target, list(module.TensorGroup), source, phase
        )
        module._formatter_profile_observed = True

    if fullname in sys.modules:
        attach(sys.modules[fullname])
        return

    class Loader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec) if hasattr(self.original, "create_module") else None

        def exec_module(self, module):
            self.original.exec_module(module)
            attach(module)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != fullname:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path, target)
            assert spec is not None and spec.loader is not None, "Stock model import unavailable"
            spec.loader = Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, Finder())
