# Compute-kernel API ordering checker (prototype)

Prototype for [#62681](https://github.com/tenstorrent/tt-metal/issues/62681): a
hardware-free check of the stateful part of the compute kernel API.

`check.py` reads `rules.yaml` and reports, per kernel and per function scope:

1. an operation with no preceding matching init;
2. a block init invalidated by `reconfig_data_format` / `pack_reconfig_data_format`
   between the init and its operation;
3. an init that is never paired with its required uninit.

Every rule in `rules.yaml` cites the operation pair it encodes. Accepted inits are
listed per operation, so aliases (`binary_tiles_init` for `add_tiles` / `mul_tiles`,
`mm_init` for `matmul_tiles`) do not produce false positives.

## Usage

```bash
python3 tt_metal/tt-llk/infra/compute_api_order/check.py --root .          # whole repo
python3 tt_metal/tt-llk/infra/compute_api_order/check.py path/to/kernel.cpp
python3 tt_metal/tt-llk/infra/compute_api_order/check.py --root . --quiet # counts only
```

Exit code 0 = no findings, 1 = findings. Nothing here is wired into CI yet.

## Tests

```bash
python3 -m pytest tt_metal/tt-llk/infra/compute_api_order/tests -q --noconftest
```

`--noconftest` skips the repository-level `conftest.py`, which needs numpy. Every
rule has a positive kernel in `tests/kernels/`; the negative kernels show that the
alias handling and the reconfig/uninit rules stay silent on correct code.

## Corpus run (this prototype, 2026-11-08)

```
$ python3 check.py --root . --quiet
scanned 1041 kernels, 61 with findings, 170 findings
  missing reduce_init(): 90
  missing matmul_init/mm_init(): 45
  missing add_init/binary_tiles_init(): 19
  missing mul_init/binary_tiles_init(): 8
  never paired with tilize_uninit(): 2
  missing mul_init/binary_tiles_init/mul_reuse_dest_init(): 2
  missing add_init/binary_tiles_init/add_reuse_dest_init(): 1
  missing tilize_init/tilize_init_short(): 1
  missing untilize_init(): 1
  missing sub_init/binary_tiles_init/sub_reuse_dest_init(): 1
```

Triage of a sample: the `matmul_init` and `reduce_init` findings are dominated by
inits hoisted into a helper function or a caller, which this prototype does not
follow — the single largest false-positive class. The tilize/uninit findings are
worth a look: 7 of 57 kernels that call `tilize_init` never call `tilize_uninit`.

## What is missing before this could be a real check

- call-graph following across helper headers and callers (`ttnn/cpp/ttnn/kernel_lib/`);
- per-thread state (UNPACK / MATH / PACK), since one source is compiled three times;
- compile-time argument comparison between an init and the operation it precedes;
- `if constexpr` / `#ifdef` path enumeration;
- agreement test against the LLK Sanitizer's operation model
  (`tt_metal/tt-llk/tools/include/sanitizer/operation.h`).
