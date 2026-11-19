# Selected default and measured reservation

The final default selects `all_roles_single_storage`:506,757,120 added bytes/rank
and31,710,200,640B resident reservation. Both final32769-token smokes pass:
`final_mesh_l1_long32769.json` and `final_mesh_l3_long32769.json`.
`doc/context_contract.json` now records that selection with unchanged262144
context, BF16 KV,1GiB default/2GiB stress planning envelopes and bounded IO.
The calculations below retain alternative scenarios for comparison; their
original pending fields describe arithmetic generation, not current selection.
Independent stage review remains pending.

# Stage5 context memory plan

All figures below are **bytes per rank**, including eight-bank padding. They
are source-derived weight allocations, not measured full-model high-water.
The four-rank model has 48 layers: 36 GDN and 12 QSA. GR injection occurs twice
in every layer, so its multiplier is **96**, not 48. Shared gate/up occurs in
all 48 layers; shared down and the scalar gate remain unchanged.

| Role | Copies | Native bytes/copy | DRAM bytes/copy | All-layer delta |
|---|---:|---:|---:|---:|
| GR injection, BF16 | 96 | 655,360 | 5,242,880 | 440,401,920 |
| GDN compact ZAB, BF16 | 36 | 8,028,160 | 9,175,040 | 41,287,680 |
| Shared split gate/up, BFP8 | 48 | 870,400 | 1,392,640 | 25,067,520 |
| Shared split decode **plus packed rows**, BFP8 | 48 | 870,400 | 2,785,280 | 91,914,240 |
| QSA packed QGate/K/V, BFP8 | 12 | 9,748,480 | 9,748,480 | 0 |
| QSA WO, BFP8 | 12 | 4,177,920 | 4,177,920 | 0 |

The two shared rows are alternatives. Both phases must be configured to release
the native weights. A decode-only policy retains native storage: for example,
GR then adds 503,316,480 bytes across the stack, and QSA adds 116,981,760 bytes.
The JSON includes each such alternative explicitly.

DRAM width sharding reserves at least 32 columns on each of eight banks. That
turns GR's one logical output into 256 physical columns instead of its native
32. This accounts for most of the persistent increase. Cores, readers, K-block,
activation dtype, and fidelity do not change the immutable DRAM weight sizes
at fixed weight dtype and packing. Policies with equal `(dtype, packed)` share
prepared weights inside each projection group. Split decode and packed rows
have different keys and retain both representations; their memory cannot be
counted once even though either representation alone consumes 1,392,640 bytes.

Compact GDN concatenates logical Z1536/A12/B12 into a single width1560 before
padding to1568. The earlier three-output candidate pads A and B separately,
giving width1600. Both round to1792 columns across eight banks and therefore
have the **same persistent bytes**. The compact candidate removes intermediate
output slicing/concatenation costs; it does not save DRAM weight bytes relative
to that earlier packed candidate.

The baseline reservation is 31,203,443,520 bytes/rank. Conservative prospective
reservation targets retain its entire existing static allowance:

| Pending scenario | Weight delta | New reservation target | Free after 1 GiB runtime | Free after 2 GiB runtime |
|---|---:|---:|---:|---:|
| All five roles, one shared representation | 506,757,120 | 31,710,200,640 | 1,394,788,544 | 321,046,720 |
| All five roles, shared split decode + packed rows | 573,603,840 | 31,777,047,360 | 1,327,941,824 | 254,200,000 |

The existing authoritative contract contains **both** a1-GiB default runtime
planning envelope and a2-GiB stress envelope. Its previously quoted827,803,840
bytes of headroom is after the2-GiB envelope. Both envelopes remain unchanged;
the1-GiB column above is an additional diagnostic, not an implicit reduction.
Use the2-GiB column for conservative carry-forward:321,046,720 bytes remain
with one shared representation,254,200,000 with split decode plus packed rows.

At logical32769, two resident residual slabs occupy335,872,000 bytes/rank.
Adding these slabs **and a separate2-GiB runtime envelope** exceeds the budget
by14,825,280 bytes in the first scenario and81,672,000 bytes in the second.
Thus the earlier1-GiB-plus-resident-IO positive margin must not be reported as
proof that a full32769 two-slab input/output allocation retains2GiB of extra
headroom. The unchanged future-stack requirement is bounded IO; arithmetic
allows at most31,328 or24,800 rows respectively in two such slabs after the
separate2-GiB envelope. These are IO budget calculations, not new public-input
or autoregressive-context limits and not measured fragmentation boundaries.

The long harness constructs the representative model/state, adds filler until
`--reserve-bytes` reaches the resident/static/state target, **then** allocates
inputs/outputs and runs the model. It does not additionally allocate a1-GiB or
2-GiB runtime block. Passing that harness proves representative execution with
resident-budget pressure; it does not measure a complete stack's runtime reserve
or guarantee2GiB remains unused after full resident IO. The revised target must
remain31,710,200,640 or31,777,047,360; do not subtract runtime reserve from it to
make the smoke fit. Separate bounded-stack IO plus the unchanged2-GiB envelope
is the context arithmetic contract. Actual final-default reservation evidence
is still required. Setup overlap, upload staging, and kernel allocations remain
separate; per-group native-plus-prepared overlap is recorded in the JSON.

KV remains BF16 with unchanged shape/page mapping, 3,221,225,472 bytes/rank for
all12 QSA layers at262144. No BFP8 KV savings are included. The existing
262144 autoregressive context and bounded-stack-IO requirement remain unchanged.
No new maximum or full-model implementation is claimed. Final selection should
update the context contract with the chosen weight delta and reservation target,
then attach the32K–64K final-default smoke evidence. A selected KV dtype/layout
change would instead trigger the maximum-context check.

`final_context_memory_plan.json` contains a pending context-contract section;
The original arithmetic pass did not mutate `doc/context_contract.json`; the owner subsequently inserted the selected section above. Generate the
arithmetic again, without TTNN imports or hardware access, using:

```bash
python -m models.autoports.qwen_qwen3_8_flash_next.tests.plan_optimized_multichip_memory
```

The tool AST-executes the actual `weight_memory_plan` and config functions from
`tt/dram_projection.py`; role shapes/counts are cross-checked against the current
constructors and pinned text config. It does not reimplement the byte formula.
