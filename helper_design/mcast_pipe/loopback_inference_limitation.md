# mcast `Pipe` — the loopback-to-self inference limitation (Round 2)

**TL;DR:** Whether a sender that sits *inside* its broadcast box also receives its own
broadcast (INCLUDE_SRC loopback) vs. does not (EXCLUDE_SRC) is a **use-case property that
geometry + the active/recipient count cannot infer**. Round 2 chose to keep the helper
knob-free and leave the one kernel that needs the un-inferable case
(`activation_reader_width_sharded`, conv width-sharded) on the **raw API**.

## The two signatures that collide

The Pipe infers loopback as: `loopback iff sender_in_rect_() && num_active_cores == area()`.
Two real kernels defeat any geometry-based rule because they are **identical to the Pipe**:

| kernel | sender in box? | recipients (`num_active`) vs box `area()` | mode it needs | why |
|---|---|---|---|---|
| matmul in0 sender | **yes** (box corner) | `num_active < area` (15 vs 16) | **EXCLUDE** | already holds the in0 block as its mcast *source*; a self-write would corrupt it |
| conv width-sharded act sender | **yes** | `num_active < area` (`num_reader_cores` < bounding-box) | **INCLUDE** | round-robin self-gather: the broadcast must land in the sender's own `act_cb`, which it then consumes |

Both present as `sender_in_box && num_active < area`, yet need **opposite** modes. The only
real distinguisher is "does the sender consume its own broadcast?" — which is data-flow
intent, not geometry. (The WS factory sets the mcast rect to the *full core bounding box*
— `conv2d_op_width_sharded_program_factory.cpp:347-348` — while only `num_reader_cores` cores
are recipients, so `num_active < area` even though the sender IS a recipient.)

## What the helper CAN express

- **EXCLUDE** in every form: sender outside the box; or sender in the box but not a recipient
  (`num_active < area`, e.g. matmul in0, conv-2d weights, ln, topk). ✓
- **INCLUDE** only when the recipients **fill the box** (`num_active == area`) — a full-box
  self-broadcast (the unit-test F3 shape). ✓

## What it CANNOT express → stays raw

- **Partial-box self-gather**: sender is a recipient but the recipients do **not** fill the
  bounding box (`num_active < area` *and* loopback wanted). Only `activation_reader_width_sharded`
  hits this. It is reverted to the raw object-API mcast and reported as a coverage gap.

## Why no knob (Round 2 decision)

Adding a `SELF_RECEIVES` boolean would migrate conv-WS too, but the user's Round-2 directive is
"do not expose a multicast mode." A `self_receives` flag is a mode in spirit, so the chosen
resolution is: **keep the helper knob-free, keep conv-WS on raw API, document the gap here.**
If a future round wants conv-WS (or other partial-box self-gather kernels) on the Pipe, the
minimal change is a single `SELF_RECEIVES` use-case template flag (default false), consulted
only when `sender_in_rect_()` — see this file for the rationale.

## Status
- conv-WS (`activation_reader_width_sharded`) reverted to raw (pre-`967cb69b01a`); raw test green.
- All other 11 migrated kernels use the knob-free inference correctly.
