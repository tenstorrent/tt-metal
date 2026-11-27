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

## TODO

```text
TODO(#12345): Remove the fallback after Blackhole firmware exposes X.
```
