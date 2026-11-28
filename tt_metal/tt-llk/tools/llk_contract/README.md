# LLK contract (Blackhole)

A machine-checkable contract between the LLK and the layers above it (compute API 1.0 and 2.0), written as a YAML table, with tools that check it against the LLK code and against itself.

Code: tt-metal main 82049e5 (27 November 2026). Covered so far: configure; tilize in full (init, execute, uninit on all three threads, and the srcA reconfigure in `tilize_init_short_with_dt`); matmul. `docs/tilize_contract.md` is the tilize contract as a document to review entry point by entry point.

## The model

Each **LLK entry point** (configure, init, execute, uninit of an operation) is described on its own by three lists:

| List | Meaning |
| --- | --- |
| `requires` | the state the call relies on when it starts |
| `ensures` | the state it leaves when it returns |
| `may_change` | fields it may leave with any value |

Everything an entry point does not mention, it leaves unchanged. The contract has no rules on call order: whether a kernel calls entry points in a working order follows from these lists, and checking order on the chip is the Sanitizer's job. The contract covers correctness, not performance. A MOP programmed in init (init ensures it, execute requires it) and one programmed in execute (execute ensures it, requires nothing) are both correct.

The **state** is listed per functional unit (`unpack`, `math`, `pack`) and named by what it does, e.g. `unpack.Tileize_mode`. A field says only what it is and how to find it in the code. Who relies on it, and who may leave it with any value, is in the entry points' lists.

A field that is a configuration register names it in `regs` (the `cfg_defines.h` name). A field that is not a register (MOP, replay buffer, ADDR_MOD slots, LLK variables) has `detect`, a pattern that finds the code setting it. A field written by another thread than its unit's says so in `thread` (the unpack thread writes the math unit's `Src_unsigned`). Where a field exists once per unpacker, `regs` gives one register for each (`srcA`, `srcB`). Each operation says which unpacker carries which operand: tilize's `in0` is on srcA, while matmul's `in0` is on srcB, because matmul needs `compute_kernel_hw_startup<SrcOrder::Reverse>`.

Two more sections:

- `invariants`: relations between fields that hold at every entry-point boundary, for state kept in two places. For example, the unpacker's config-context register and the LLK's copy of it in `unp_cfg_context` must agree. An entry point that changes one changes the other.
- `kernel_start`: what is assumed before the first entry point, for state no configure sets.

**Where each entry point is in the code** is kept apart from the contract, in `contract/llk_bindings.yaml`: the LLK function that implements it on each thread (`llk`), the compute API functions that call it (`compute_api`), and the template arguments those callers pass (`template`, e.g. tilize's execute calls datacopy with `BroadcastType::NONE`; the code checks then skip the `if constexpr` branches those arguments rule out). The contract says what an entry point requires and ensures; the bindings say where that is in the code, so renaming a function changes the bindings, not the contract.

Every `requires`/`ensures` item has an **ID** made from its place in the file, e.g. `tilize.execute.requires:unpack.Tileize_mode`. All tools report by ID. An entry point may give the reason for an item in `why`.

Example, shortened from `contract/llk_contract.yaml`:

```yaml
tilize:
  operands:
    in0: {unpacker: srcA, layout: row_major}
  init:
    requires: {}
    ensures:
      unpack.Tileize_mode: 1
      unpack.MOP: "tilize(in0, op.ct_dim, op.narrow_tile) if not is_8bit(in0.format.l1)"
  execute:
    requires:
      unpack.Tileize_mode: 1
      unpack.MOP: "tilize(in0, op.ct_dim, op.narrow_tile) if not is_8bit(in0.format.l1)"
    why:
      unpack.Tileize_mode: "the unpacker rearranges row-major input into faces only in tileize mode"
  uninit:
    ensures:
      unpack.Tileize_mode: 0
```

## Folder

```
llk_contract/
├── README.md
├── run_checks.sh             runs every check against a tt-metal checkout
├── contract/                 the contract itself
│   ├── llk_contract.yaml           state, invariants, kernel-start assumptions, entry points
│   ├── llk_contract.schema.yaml    the format of llk_contract.yaml (JSON Schema, written as YAML)
│   ├── llk_bindings.yaml           where each entry point is in the code: LLK functions, compute API callers
│   └── llk_bindings.schema.yaml    the format of llk_bindings.yaml
├── tools/                    the checks
│   ├── contract_tool.py            validate, check-code, check-api, trace, check-sequence
│   ├── contract_z3.py              proofs with Z3 for every format (prove, prove-sequence)
│   ├── contract_values.py          evaluates values for concrete operands (used by check-sequence)
│   ├── llk_cfg_map.py              which registers each LLK function writes (used by check-code)
│   ├── compute_api_map.py          which LLK functions each compute API function reaches (for check-api)
│   └── llk_source.py               reading the LLK source as text, shared by the tools
├── examples/sequences/       example call sequences for check-sequence and prove-sequence
└── docs/                (written by run_checks.sh)
    ├── tilize_contract.md          the tilize contract, entry point by entry point, with code lines (for review)
    └── requirements_trace.md       every requirement, its reason and the code lines that meet it
```

`llk_contract.yaml` is the contract; `llk_bindings.yaml` ties it to the code. The rest checks them.

## Checks and results

| Command | Checks | Result |
| --- | --- | --- |
| `validate` | The file follows the schema. Every field and index exists. An entry point that changes one field of an invariant changes the others. Plus the properties below. | 0 errors, 4 findings |
| `check-code` | The LLK functions named for each entry point set every field it ensures. Anything they set that the entry point does not mention is reported. This is the check for CI. | 118 of 118 set; 0 writes not mentioned |
| `check-api` | Each compute API function listed reaches the LLK function named for each thread. | 32 of 32 |
| `trace` | Every requirement by ID with its reason. For each `ensures`, the file and line where the LLK sets it. With `--op`, one operation as a review document. It also lists, derived from the entry points, the state some entry point relies on (what a runtime checker such as the Sanitizer has to track) and the state none relies on. | 188 requirements; 118 of 118 ensures traced; 50 fields relied on, 26 not |
| `check-sequence` | A sequence of calls, for given operands, without a chip. Before each call, are its requirements met? Each failure is reported by ID, with the call that last set the field. | 2 correct example sequences; 3 with failures (10, 17, 13) |
| `contract_z3.py prove` | The contracts compose for every format, tile shape and parameter (below). | see below |

`validate` reports four properties of the contracts:

- **establish**: every field an execute requires is ensured, with the same value, by its own init, by configure, by the execute itself, or at kernel start.
- **restore**: what an init changes away from configure's value, and whether its uninit puts it back.
- **interfere**: operation A leaves a field changed that B's execute takes from configure.
- **between**: B's execute takes a field from B's init, and entry point A changes it. A called between B's init and B's execute therefore needs B's init again.

None of these is a rule on order. They follow from the lists.

### Proofs for every format: `contract_z3.py`

`validate` compares values as text. `contract_z3.py` translates the contract to Z3 and decides each property for every Blackhole data format of every operand, `face_r_dim` 1, 2, 4, 8 or 16, `num_faces` 1, 2 or 4, and every parameter value. For each requirement, Z3 either proves it or gives a counterexample: values for which it fails.

Format predicates, sizes, conditions and arithmetic are translated as the LLK computes them. Every other function in a value (`canonical_z_stride`, `matmul(op)`, `datacopy`) is uninterpreted: Z3 knows only that the same function of equal arguments is equal. A proof therefore holds whatever those hardware functions compute.

| Property | Sequence | Result |
| --- | --- | --- |
| establish | configure, B.init, B.execute | tilize PROVED (39 requirements); matmul PROVED (31) |
| restore | configure, B.init, B.uninit | tilize PROVED; matmul leaves `Haloize_mode` set when `transpose=1` |
| after | configure, A.init, A.execute, A.uninit, B.init, B.execute | tilize then matmul PROVED; matmul then tilize PROVED |
| between | configure, B.init, A.entry, B.execute | the requirements that fail, with the values for which they fail |

`prove-sequence` proves a sequence file for all operands. Both correct example sequences are PROVED.

### Sequences: `check-sequence`

A sequence file lists the calls, the operation parameters for each, the kernel settings, and what the kernel configured on each unpacker and its output (formats, tile shape). Values are computed for those operands (`contract_values.py`). Format predicates are evaluated as the LLK evaluates them (`is_8bit` as `IS_8BIT_FORMAT`). Hardware functions stay terms, and the same function of the same arguments is the same value. A value that cannot be computed is reported as *undecided*, with the reason, never as a failure.

From `examples/sequences/matmul_init_tilize_init_matmul_execute.yaml`:

```
   4. matmul.execute           10 not met
        - not met: matmul.execute.requires:unpack.Tileize_mode = 0: is 1, set by call 3 (tilize.init.ensures:unpack.Tileize_mode)
        - not met: matmul.execute.requires:unpack.Num_faces[in1] = 4: is 1, set by call 3 (tilize.init.ensures:unpack.Num_faces[in0])
        - not met: matmul.execute.requires:unpack.MOP = matmul(op): is tilize(srcA, op.ct_dim, op.narrow_tile), set by call 3 (tilize.init.ensures:unpack.MOP)
        ...
```

## Findings

1. **Haloize_mode after matmul.** Matmul's init sets `Haloize_mode` to the transpose flag, and its uninit (empty) leaves it set. Every other unpack init sets it again, so no operation is affected.
2. **Tilize and matmul share program state.** They share the MOPs, the unpacker x_end, `Haloize_mode` and math ADDR_MOD slots 0, 2 and 3. Matmul's replay buffers and slots 1, 4 and 5 are not touched by tilize. Z3 states the conditions exactly. For example, tilize's init between matmul's init and execute breaks `Num_faces` only for tiles with more than one face, and `Haloize_mode` only when matmul transposes.
3. **Matmul's ADDR_MOD slots.** Matmul's init programs ADDR_MOD slots 0 to 5. With the row MOP or the throttled MOP, its execute also programs slot 6 at entry and restores it before returning.
4. **ADDR_MOD 6 and 7 are now invariants.** Slots 6 and 7 used to be programmed by matmul's init and by some SFPU inits alike, so an SFPU init between a throttled matmul's init and execute changed a slot matmul relied on. Since #63389 and #63390 (merged on 26-27 November) configure sets them to fixed values on the math and pack threads (`llk_math_common.h:29-41`, `llk_pack.h:468-490`), and "any op that reprograms either slot must restore its invariant before returning". The contract states exactly that: configure ensures them, matmul's execute ensures them again. `check-code` caught this change: on the commit before it reported matmul's init no longer setting slot 6 and configure setting slots 6 and 7 that the contract did not mention.
5. **`mm_uninit` on Blackhole.** It calls no LLK function, because it is Quasar-only (`matmul.h:148-152`), so the LLK matmul uninits are not reached from the compute API.
6. **Tilize puts the packer in Tilize mode, and its uninit takes it out.** On Blackhole, `tilize_init` arms the packer in Tilize mode (its own MOP, replay buffer, ADDR_MODs and a Z stride two faces long for full-width tiles), except for 8-bit inputs. `tilize_uninit` re-inits the packer in Default mode (`tilize.h:241-243`). An op that packs after tilize without its own pack init relies on that uninit; Z3 proves tilize's uninit restores every field its init changed.
7. **Datacopy relies on a zero flag it does not set.** Tilize's math execute (datacopy) relies on the zero flag, which neither its init nor its execute sets. It holds configure's default unless an earlier op changed it (marked TO CONFIRM in the contract).
8. **The compute API switches the config context.** `api/compute/experimental/sdpa_weighted_reduce.h:139` switches the unpacker config context itself. It does so through the LLK helper, so the invariant holds.

## Limits

- `check-code` checks that the code writes a field, not the value it writes. Checking values would need each entry point's code translated to Z3 as well.
- The tools read code as text and follow every branch, except `if constexpr` branches ruled out by an entry point's `template` in the bindings. A field set on only some paths counts as set. Thread handoff (`math_wait_for_dest_available`, `dest_section_done`) is not covered.
- Sequences and Z3 take each unpacker to carry one buffer for the whole sequence, so a reconfigure to another buffer is checked on its own, not across calls.
- `trace` and the write check work at register granularity. Fields that share a register (the tile descriptor holds format, x dim and face count) list the same lines. A field in the same config word as a mentioned field is not reported as not mentioned.
- The Z3 domain is wider than real kernels: any pair of L1 and unpacked formats, and any parameter value not listed in `supports`. A counterexample can therefore be a combination no kernel uses, while a proof holds for all, real ones included. Filling in `supports` narrows the domain.

## Open questions

1. **Kernel start.** What holds before any configure (`kernel_start`)? For example, `CLR_DVALID_SrcA_Disable` is set by no configure.
2. **`Int8_math` in one-sided reconfigs.** Configure and the two-sided reconfig set `INT8_math_enabled` from SrcA or SrcB (`llk_math_common.h:45,275`). The one-sided reconfigs use only the operand they change (`llk_math_common.h:206,240`). After `reconfig_srca(Float16_b)` with an Int32 SrcB, the flag is 0, while configure would set it to 1. Is this intended?
3. **Tilize's zero flag.** Tilize's math path relies on the zero flag, which datacopy's init does not set. The compute API `copy_init` sets it (`tile_move_copy.h:69`). Which value does datacopy need?
4. **Supported formats and tile shapes** per operation (`supports`, TO CONFIRM).
5. **Coverage.** The other operations are not in the table yet, and matmul's tile-size GPRs (`TILE_SIZE_A/B`, set by its init and read by its execute) are not yet fields.

## Run

The folder lives in tt-metal at `tt_metal/tt-llk/tools/llk_contract/`. From there:

    pip install pyyaml jsonschema z3-solver
    ./run_checks.sh                 # checks the tt-metal checkout it is in
    ./run_checks.sh ~/tt-metal      # or another checkout

or one check at a time, from this folder (`~/tt-metal` is the checkout):

    python3 tools/contract_tool.py validate       contract/llk_contract.yaml
    python3 tools/contract_tool.py check-code     contract/llk_contract.yaml ~/tt-metal
    python3 tools/compute_api_map.py ~/tt-metal --all --json api_map.json
    python3 tools/contract_tool.py check-api      contract/llk_contract.yaml api_map.json
    python3 tools/contract_tool.py trace          contract/llk_contract.yaml ~/tt-metal --md docs/requirements_trace.md
    python3 tools/contract_tool.py trace          contract/llk_contract.yaml ~/tt-metal --op tilize,reconfig_srca --md docs/tilize_contract.md
    python3 tools/contract_tool.py check-sequence contract/llk_contract.yaml examples/sequences/tilize_ok.yaml
    python3 tools/contract_z3.py prove            contract/llk_contract.yaml
    python3 tools/contract_z3.py prove-sequence   contract/llk_contract.yaml examples/sequences/tilize_ok.yaml
