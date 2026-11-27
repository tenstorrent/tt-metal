"""Checks that the oracle of g6_clone is the function the reference implementation computes.

    python3 -B check_oracle.py
        prints one line per check and writes oracle_report.json

Groups: 1 equivalence with the PyTorch function executed on this host (E1), 2 probes (bit
patterns per production dtype, special values, shapes of the call matrix, new storage),
3 reformulations (none: math.md derives no F-id, a copy has no equivalent form that changes
values), 4 precision (storage only: a copy charges nothing to any cell).
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True

from oracle_kit import Checks, seeded  # noqa: E402
from pytorch_g6_clone import _BITS_VIEW, pytorch_g6_clone, reference  # noqa: E402

checks = Checks("g6_clone")
EVIDENCE = "E1"
SHAPES = [(1, 1, 32, 32), (2, 3, 64, 96), (1, 2, 17, 50), (16,), (1, 1), (3, 5, 7)]
HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = os.path.join(HERE, "..", "sources", "local", "scenarios.json")
DTYPES = {"BFLOAT16": torch.bfloat16, "FLOAT32": torch.float32, "INT32": torch.int32, "UINT32": torch.uint32}


def upstream(input_tensor, **kwargs):
    """torch.clone of the installed torch 2.11.0, the build pinned under sources/ (E1)."""
    return torch.clone(input_tensor)


def bits(tensor):
    view = _BITS_VIEW.get(tensor.dtype)
    return tensor.view(view) if view is not None and view != tensor.dtype else tensor


def bit_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b))


def draw(shape, dtype, generator):
    """Random bit patterns of the dtype: every pattern, NaN payloads and subnormals included."""
    width = {torch.bfloat16: 16, torch.float32: 32, torch.int32: 32, torch.uint32: 32}[dtype]
    raw = torch.randint(-(2 ** (width - 1)), 2 ** (width - 1), shape, generator=generator,
                        dtype=torch.int64)
    if width == 16:
        return raw.to(torch.int16).view(torch.bfloat16)
    as_int32 = raw.to(torch.int32)
    return as_int32 if dtype == torch.int32 else as_int32.view(dtype)


# 1. equivalence (float64, the definition against the executed built-in)
checks.evidence = EVIDENCE
checks.scope = ""
for index, shape in enumerate(SHAPES):
    x = torch.randn(shape, dtype=torch.float64, generator=seeded(index))
    checks.equal(f"upstream {shape}", reference(x), upstream(x))

# 2. probes
# 2a. every production dtype, random bit patterns: reference, oracle and upstream bit-identical
for index, (name, dtype) in enumerate(DTYPES.items()):
    x = draw((3, 5, 37), dtype, seeded(200 + index))
    checks.true(f"bit patterns {name}: reference equals upstream", bit_equal(reference(x), upstream(x)))
    checks.true(f"bit patterns {name}: oracle equals reference", bit_equal(pytorch_g6_clone(x), reference(x)))

# 2b. special values survive: -0.0, +-inf, NaN with payload, subnormals, extremes
specials_f32 = torch.tensor([0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0xFFA00000,
                             0x00000001, 0x807FFFFF, 0x7F7FFFFF], dtype=torch.int64).to(torch.int32).view(torch.float32)
specials_bf16 = torch.tensor([0x8000, 0x7F80, 0xFF80, 0x7FC1, 0xFFA0, 0x0001, 0x807F, 0x7F7F],
                             dtype=torch.int64).to(torch.int16).view(torch.bfloat16)
extremes_i32 = torch.tensor([-2 ** 31, 2 ** 31 - 1, -1, 0, 2 ** 24 + 1], dtype=torch.int32)
extremes_u32 = torch.tensor([0, 2 ** 32 - 1, 2 ** 31, 2 ** 24 + 1], dtype=torch.int64).to(torch.int32).view(torch.uint32)
for name, x in (("float32", specials_f32), ("bfloat16", specials_bf16), ("int32", extremes_i32),
                ("uint32", extremes_u32)):
    checks.true(f"special values {name} kept bit for bit", bit_equal(reference(x), x) and bit_equal(pytorch_g6_clone(x), x))

# 2c. a float32 round trip would not be a copy: int32 2^24 + 1 is lost through float32
checks.true("float32 round trip loses int32 2^24 + 1 (why the oracle keeps the dtype)",
            int(torch.tensor([2 ** 24 + 1], dtype=torch.int32).to(torch.float32).to(torch.int32)[0]) != 2 ** 24 + 1)

# 2d. new storage: writing the result leaves the input unchanged
x = torch.randn(4, 8, generator=seeded(300))
before = x.clone()
out = reference(x)
out.fill_(7.0)
checks.true("new storage: writing the result leaves the input unchanged", torch.equal(x, before)
            and out.data_ptr() != x.data_ptr())

# 2e. every case of the call matrix (sources/local/scenarios.json): output shape and dtype equal
#     the input's, and reference, oracle and upstream agree bit for bit on a random draw
with open(SCENARIOS) as handle:
    cases = json.load(handle)["cases"]
same_meta = all(c["output"]["shape"] == c["tensor_args"][0]["shape"] and c["output"]["dtype"] == c["tensor_args"][0]["dtype"]
                and c["output"]["layout"] == c["tensor_args"][0]["layout"] and c["output"]["memory"] == c["tensor_args"][0]["memory"]
                and not c["literal_args"] and not c["kwargs"] and len(c["tensor_args"]) == 1 for c in cases)
checks.true(f"call matrix: {len(cases)} cases, one tensor each, no other argument, output metadata equals input", same_meta)
all_ok = True
for index, case in enumerate(cases):
    spec = case["tensor_args"][0]
    x = draw(tuple(spec["shape"]), DTYPES[spec["dtype"]], seeded(400 + index))
    all_ok &= bit_equal(reference(x), upstream(x)) and bit_equal(pytorch_g6_clone(x), x)
checks.true(f"call matrix: all {len(cases)} production shapes bit-identical (reference, oracle, upstream)", all_ok)

# 3. reformulations: none (math.md derives no F-id)

# 4. precision (storage only: rows with status "info", never a pass or a failure)
x = torch.randn(SHAPES[1], generator=seeded(100))
for dtype in ("float32", "bfloat16"):
    checks.cell(f"{dtype} cell, randn", pytorch_g6_clone, (x,), dtype)

checks.finish()
