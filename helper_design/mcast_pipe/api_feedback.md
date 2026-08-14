# `mcast_pipe` open API feedback

This file contains only unresolved API questions. Once an item is implemented,
rejected, or otherwise closed, remove it here; durable decisions belong in
`changelog.md`. Use `design/api_feasibility.md` when resolving an item depends
on the production inventory or helper capability analysis.

## API-002 — Enforce the kernel's permitted sender/receiver face

- **Date:** 2026-08-04
- **Status:** Open
- **Surface:** host mcast wire and `dataflow_kernel_lib::McastArgs`
- **Trigger:** `activation_reader_width_sharded.cpp` constructs both
  `act_mcast_args.sender(noc)` and `act_mcast_args.receiver(noc)`. That is valid
  for its rotating protocol, but it raises the split-kernel case: what prevents
  a receiver-only kernel from constructing a sender pipe, or vice versa?

### Current behavior

The CT wire does not describe which face a kernel is allowed to construct.
Consequently, both methods are always available:

```cpp
auto sender = args.sender(noc);
auto receiver = args.receiver(noc);
```

For a fixed sender, the four RT words are a destination rectangle. For a fixed
receiver, the same four slots contain `[sender_x, sender_y, 0, 0]`. A mistaken
`sender()` call in a receiver-only kernel therefore interprets receiver data as
a rectangle; there is no compile-time diagnostic.

### Desired contract

The host binding should declare the permitted face for each kernel:

- **SenderOnly**
- **ReceiverOnly**
- **Both** — required for rotating protocols such as width-sharded Conv and for
  any single binary that legitimately performs either role.

This is host-known compile-time metadata and should ride on the CT wire rather
than become another caller-supplied `McastArgs` template argument. With a
constexpr face mask, `sender()` and `receiver()` should reject use of a face not
permitted by the kernel binding, preferably with `static_assert` when that
method is instantiated.

### Scope

This feedback tracks only compile-time sender/receiver-face enforcement. Keep
the current uniform CT/RT block sizes and role-neutral runtime layout. `Both`
permits construction of both faces; per-core runtime-role validation and RT
wire compaction are not part of this item.

### Resolution checklist

- Inventory sender-only, receiver-only, and Both kernels.
- Encode the permitted face in the existing self-describing CT metadata without
  changing the current block widths.
- Add negative compile tests for calling `sender()` on ReceiverOnly and
  `receiver()` on SenderOnly.
- Re-run the helper suites and all migrated production inventories.
