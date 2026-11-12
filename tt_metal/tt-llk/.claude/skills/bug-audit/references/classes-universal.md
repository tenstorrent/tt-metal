# Universal bug classes: the floor for every audit

These classes apply to any codebase. Every hunt covers all of them, whatever a repo pack says. A pack REORDERS them
(it says which ones this codebase actually suffers from and where) and ADDS domain classes. It never removes one.
A class with no history in a repo is still hunted: history records only the bugs someone noticed.

Use the `id` as the finding's `category` when it fits; otherwise invent a short kebab-case name.

## Arithmetic and indexing
- `index-math`: off-by-one; wrong stride or dimension; double indexing; row vs column swapped; an index computed in
  one unit and used in another.
- `integer-width`: truncation (a wide value stored in bool/uint8/uint16), signed/unsigned mix-up, overflow in size or
  offset math, a narrowing conversion hidden by `auto` or a template. Includes narrowing at a CALL BOUNDARY: a wider
  argument into a narrower parameter or constructor member, or into a packed bit field bounded by a `*_MASK` constant.
- `rounding`: `ceil` applied after integer division (a no-op); floor where ceil was meant; alignment rounded the wrong
  way; a remainder or tail element dropped.
- `units`: bytes vs words vs pages vs tiles vs elements; us vs ms; per-core vs total; a scale factor off by 2^n.

## Control flow and logic
- `inverted-condition`: wrong comparison, swapped operands, `&&`/`||` confusion, a condition that is always true or
  always false, De Morgan slips.
- `missing-exit`: missing `return`/`break`, unintended fallthrough, a guard that logs "skipping" and then proceeds.
- `loop-bounds`: a loop that runs zero times or once too often; an accumulator never reset, or reset too often; a loop
  variable shadowed.
- `dead-guard`: an assert or check that cannot fire: compiled out in release builds, checks the wrong variable,
  runs after the damage, or compares against itself.
- `missing-case`: a switch or if-chain that misses an enum value or type that sibling code handles; a default that
  silently does the wrong thing.
- `lossy-compare`: an equality or range test done after a mask, cast or truncation that merges distinct values (for
  example `mask(x) == K` when several enum values mask down to K).

## Copy-paste and binding
- `copy-paste`: wrong variable, member, enum, template argument or lookup key in a duplicated block. The strongest
  signal is a near-identical sibling that differs in exactly one token.
- `arg-binding`: an argument binding silently to the wrong parameter (defaults, overloads, bool/int conversions,
  swapped same-typed parameters, positional template args after a signature change).
- `stale-sibling`: a fix or signature change applied to one variant (arch, dtype, layout, overload) but not to its
  siblings.
- `sibling-divergence`: a call site, formula or constant that differs from all its siblings. Examples: the same API
  called everywhere else with an extra argument; the same quantity derived by two different formulas in one file;
  one register literal that disagrees with its field macro and its neighbours.
- `dead-store`: a value computed or a parameter received and then never used, which usually means a caller's
  intent was dropped (an ignored memory config, a flag overwritten before use).

## Data lifetime and state
- `lifetime`: use-after-free or use-after-move; a dangling reference to a temporary or a dead stack frame; an iterator
  or pointer invalidated by a container resize.
- `uninitialized`: a read of an uninitialized variable, member or buffer, including padding that later becomes
  visible.
- `empty-access`: `.begin()`, `front()`, `[0]`, `->first` or `value()` on a container or optional that can be
  empty on some platform, configuration or input (harvested parts, simulators, single-chip, zero-size).
- `container-misuse`: pre-size plus append (the container ends up double length); reserve vs resize confusion; a
  missing clear between uses.
- `stale-state`: state (a config, cache or register) left over from a previous call and not reset; an init/uninit
  pair out of balance.
- `resource-leak`: a leaked handle, fd, allocation or lock on an early return or an exception path.

## Caching and identity
- `cache-key`: a cache, memo or hash key that omits a field which changes the result, so a stale entry is reused for
  a different configuration.
- `identity-collision`: two distinct objects that map to the same name, id, address or slot.

## Contracts and APIs
- `contract-mismatch`: producer and consumer disagree about layout, order, count, units or meaning (host vs device,
  writer vs reader, caller vs callee, runtime args vs the kernel that reads them).
- `validation-gap`: input validation that accepts a shape, value or config the implementation cannot handle, or
  rejects one it can.
- `error-handling`: an error swallowed, a wrong error code, or a failure reported as success; cleanup skipped on the
  error path; a diagnostic that prints a bound or range different from the one its guard enforces.
- `validate-vs-impl`: a validator that admits an input combination (layout, dtype, shard spec, config) that no
  implementation path handles, or an implementation that relies on something the validator never checks.

## Concurrency
- `data-race`: shared state touched by two agents (threads, cores, processes) without the ordering its readers
  assume.
- `ordering`: a signal (flag, semaphore, credit, notification) published before the data it announces is visible.
- `deadlock`: a wait that the matching post can never satisfy on some path; a count mismatch between the two sides;
  a lock-order inversion.
- `atomicity`: a read-modify-write that another agent can interleave.

## Numerics
- `numeric-edge`: NaN, Inf, denormals, signed zero, or the largest and smallest representable values handled wrongly.
- `precision`: precision lost through an intermediate format, the wrong accumulation dtype, or a lossy conversion
  where the contract promises exactness.
- `numeric-formula`: a wrong formula or constant, or a wrong approximation range or its guard.

## Parameter space
- `param-combination`: correct at the default configuration but wrong for a specific combination of shape, size,
  count, dtype, layout, flag or platform. Enumerate what callers can actually pass, and check the edges (0, 1, odd,
  non-multiple, max, empty).
- `platform-divergence`: behaviour that differs across platforms, arches or build modes in a way the code does not
  handle.

## Tests and tooling (when the scope includes them)
- `test-blind-spot`: a test that cannot fail. It drops a parameter before the code under test sees it, compares the
  output with itself, skips on the case it claims to cover, or is gated behind a flag CI never sets.
- `build-config`: a define, flag or option that silently changes behaviour, or is not propagated to every target that
  needs it.

## Symbols and the build
- `symbol-resolution`: a name that resolves to something other than intended. Examples: a namespace whose last
  component shadows an existing reachable namespace; an unqualified lookup picking a new declaration; an extern or
  linker symbol that differs between link modes.
- `preprocessor`: a macro defined on the compile line that collides with an identifier in a header; `#if`/`#elif` on a
  macro some build flavour leaves undefined (it evaluates to 0); a hand-built define list that diverges from the
  canonical one.
- `linkage-visibility`: a symbol declared in a hidden-visibility scope, or defined out of line, but called across a
  shared-library boundary in a supported build mode; ODR or duplicate-definition hazards between libraries.
