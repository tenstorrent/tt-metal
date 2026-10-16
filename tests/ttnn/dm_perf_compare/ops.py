# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-op adapter: the three entry points of a DM op and the arguments each one takes.

`default` is the op as a user calls it (`ttnn.<op>`, no implementation forced). `native` and
`codegen` are `<op>_force_native` / `<op>_force_codegen` in `ttnn._ttnn.operations.data_movement`.
The forced entries take fewer arguments than the public one, and `codegen` raises for a call it does
not support instead of falling back; both are reported per call, never worked around.
"""

import re
import zlib

IMPLS = ("default", "native", "codegen")

# Defaults of the public entry's optional arguments. A forced entry that does not accept an argument
# is still comparable when the call passes that argument's default; any other value is a different
# call, so the forced implementation is skipped for it.
IMPLICIT_DEFAULTS = {"use_multicore": True, "sparse_grad": False}

_SIGNATURE = re.compile(r"\((.*?)\)\s*(?:->|$)", re.S)


class Refused(Exception):
    """A forced implementation cannot take this call."""


def accepted_kwargs(fn):
    """Parameter names of a nanobind function, from its signature line; None when unreadable."""
    doc = getattr(fn, "__doc__", None) or ""
    match = _SIGNATURE.search(doc.splitlines()[0]) if doc else None
    if not match:
        return None
    names = set()
    for part in re.split(r",\s*(?![^\[\]()]*[\]\)])", match.group(1)):
        part = part.strip()
        if part and part not in ("*", "/"):
            names.add(part.lstrip("*").split(":")[0].split("=")[0].strip())
    return names


def split_call_args(args, accepts):
    """Keyword arguments a forced entry takes, or raise Refused naming what it cannot take."""
    if accepts is None:
        return dict(args)
    taken, refused = {}, []
    for key, value in args.items():
        if key in accepts:
            taken[key] = value
        elif value is None or IMPLICIT_DEFAULTS.get(key, object()) == value:
            continue
        else:
            refused.append(key)
    if refused:
        raise Refused("forced entry does not take %s" % ", ".join(sorted(refused)))
    return taken


def seed_for(call_id):
    return zlib.crc32(call_id.encode())


class OpAdapter:
    def __init__(self, op, ttnn):
        self.op = op
        self.ttnn = ttnn
        module = ttnn._ttnn.operations.data_movement
        self.entries = {
            "default": getattr(ttnn, op, None),
            "native": getattr(module, f"{op}_force_native", None),
            "codegen": getattr(module, f"{op}_force_codegen", None),
        }
        self.accepts = {impl: accepted_kwargs(fn) for impl, fn in self.entries.items() if fn is not None}

    def available(self, impl):
        return self.entries[impl] is not None

    def memory_config(self, spec):
        ttnn = self.ttnn
        if spec == "DRAM":
            return ttnn.DRAM_MEMORY_CONFIG
        if spec == "L1":
            return ttnn.L1_MEMORY_CONFIG
        if isinstance(spec, dict):
            (x0, y0), (x1, y1) = spec["grid"]
            shard_spec = ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))}),
                tuple(spec["shard_shape"]),
                getattr(ttnn.ShardOrientation, spec.get("orientation", "ROW_MAJOR")),
            )
            return ttnn.MemoryConfig(
                getattr(ttnn.TensorMemoryLayout, spec["layout"]),
                getattr(ttnn.BufferType, spec.get("buffer", "L1")),
                shard_spec,
            )
        raise ValueError(f"unknown memory_config {spec!r}")

    def host_tensor(self, torch, shape, dtype, seed, randint=None):
        generator = torch.Generator().manual_seed(seed)
        if dtype in ("uint32", "int32", "uint16") or randint is not None:
            lo, hi = randint or (0, 1000)
            return torch.randint(lo, hi, tuple(shape), generator=generator, dtype=torch.int32)
        return torch.randn(tuple(shape), generator=generator)

    def device_tensor(self, torch, device, shape, dtype, layout, memory_config, seed, randint=None):
        ttnn = self.ttnn
        return ttnn.from_torch(
            self.host_tensor(torch, shape, dtype, seed, randint),
            dtype=getattr(ttnn, dtype),
            layout=ttnn.TILE_LAYOUT if layout == "tile" else ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=self.memory_config(memory_config),
        )

    def _value(self, torch, device, value, seed):
        if value in ("DRAM", "L1"):
            return self.memory_config(value)
        if isinstance(value, dict) and "tensor" in value:
            spec = value["tensor"]
            return self.device_tensor(
                torch,
                device,
                spec["shape"],
                spec["dtype"],
                spec.get("layout", "row_major"),
                spec.get("memory_config", "DRAM"),
                seed,
                spec.get("randint"),
            )
        return value

    def prepare(self, torch, device, call):
        """The call's input tensor and keyword arguments, built once and shared by every impl."""
        seed = seed_for(call["id"])
        x = self.device_tensor(torch, device, call["shape"], call["dtype"], call["layout"], call["memory_config"], seed)
        kwargs = {
            key: self._value(torch, device, value, seed + i + 1)
            for i, (key, value) in enumerate(sorted(call["args"].items()))
        }
        return x, kwargs

    def bind(self, impl, x, kwargs):
        """A zero-argument callable running `impl` on the call, or raise Refused."""
        fn = self.entries[impl]
        if fn is None:
            raise Refused(f"{self.op} has no {impl} entry")
        taken = kwargs if impl == "default" else split_call_args(kwargs, self.accepts[impl])
        return lambda: fn(x, **taken)
