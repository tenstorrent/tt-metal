# r04-b01-a03: stack r04-b04-a02's writer (posted output drain + ack-free stick push) onto the HiFi2 PRE node r04-b01-a02

## Motivation
Round 4 produced two independent wins on disjoint parts of the kernel:
- r04-b01-a02 (this node's parent, 1.3911): PRE x*x ELWMUL and ones*S^T matmul at HiFi2. Compute-only. The stat is
  ready 0.2-0.6 µs sooner after the last input lands, so the AG starts earlier.
- r04-b04-a02 (1.3997, best): writer-only. Posted drain writes (r04-b04-a01: drain -0.2..-0.46 µs) plus the
  flush-then-inc stick push (r04-b03-a01: W_PUSH 0.63 -> 0.25 µs), on top of the same streamed gamma
  (r04-b01-a01) that the parent already has.
The parent still runs the non-posted drain and the 0.63 µs push handshake. The best node still runs PRE at HiFi4.
Both lineages share r04-b01-a01's writer, so `git diff r04-b01-a02 r04-b04-a02` is exactly the two writer changes
plus the HiFi2 compute edit (see context.md). This is the parent reflection's #1 suggestion.

## Mechanism
Replace `dit_rmsnorm_fused_worker_writer.cpp` with r04-b04-a02's version. It is the parent's writer plus:
- stick push: `async_writes_flushed()` then the arrival inc on the same NoC/VC; atomic barrier at kernel end;
- output drain: `async_write<NocOptions::POSTED>` and posted flush before each pop; posted flush at kernel end.
Compute stays at the parent's HiFi2 PRE. The result is r04-b04-a02 + HiFi2 PRE, and the three-way merge is clean.

## Why this is not a repeat
No node has both HiFi2 PRE and the posted drain/ack-free push. r04-b04-a02 and r04-b03-a02 stacked the writer wins at
HiFi4. r04-b01-a02 applied HiFi2 with the old writer. This combines previously successful pieces that touch
different parts of the kernel. No new mechanism.

## Expected effect and risk
The HiFi2 effect moves the AG start (push end -0.3..-0.6 µs). The push cut moves it a further ~0.38 µs. The posted
drain shortens the post-AG tail. In r04-b04-a02 / r04-b03-a02 the AG-start and drain gains were roughly additive.
Expected h3584 ~11.6, h4096 ~13.1, h6144 ~16.4, h7168 ~17.4 µs, score ~1.43-1.44. Risk: the AG may become gated by
cross-chip arrival or fabric instead of local pushes (as on h4096 in r04-b04-a02), which would make the gains
sub-additive. Accuracy should match the parent (max_abs ~0.022-0.024). The writer change produces the same bytes.
