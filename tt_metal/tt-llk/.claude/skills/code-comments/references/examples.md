# Code comment examples

Avoid/prefer pairs for the rules in [../SKILL.md](../SKILL.md).

## Narration versus invariant

Avoid narration:

```cpp
// Set the destination address.
set_dst_write_addr<...>(0);
```

Prefer the non-obvious invariant:

```cpp
// Pack release clears this Dest half before math reuses it.
_llk_pack_dest_section_done_<...>();
```

## Name the mechanism

Avoid a vague outcome:

```cpp
// Needed for perf.
```

Prefer the mechanism:

```cpp
// Dest section release clears the packed half, so no explicit ZEROACC here.
```

## Edit history

Avoid:

```python
# The old wedge skip was removed because this works now.
```

Prefer no comment when the test name and assertion already explain the case.

## Duplicated detail

Avoid:

```cpp
// Reserve one accumulator slot, leaving capacity - 1 chunks available.
// The final accumulator is reserved for the reduction result.
```

Prefer one line:

```cpp
// Reserve the final accumulator slot for the reduction result.
```

## PR evidence

Avoid a measurement that changes with every build:

```cpp
// Adds about 400 B to the image; on one board it moved 3 of 900 counter values.
```

Prefer no comment, or the durable constraint behind the numbers:

```cpp
// Compiled only with its option: the capture does not fit the image budget otherwise.
```

## ISA link versus "measured"

Avoid restating a documented fact as an observation:

```cpp
// Measured: one data word is 64 bytes on this architecture.
```

Prefer the source:

```cpp
// Data word = one NoC flit, see the ISA NoC counters page.
```

When the ISA disagrees or is silent, say so:

```cpp
// The ISA counts this register in bytes; confirm with the ISA owners before treating it as words.
```

## Oversharing

Avoid design-internal references:

```cpp
// Same width as the counter register in the block's RTL source file.
```

Prefer the public or repository source:

```cpp
// Counter width per the ISA counters page.
```

## Rationale repeated across files

Avoid the same paragraph in a module docstring, its README, and the doc page:

```python
# Kept in its own module because its keys follow a different naming scheme.
```

Prefer it once, in the README, and nothing in the module when its name and
imports already make the split obvious.

## TODO

```text
TODO(#12345): Remove the fallback after Blackhole firmware exposes X.
```
