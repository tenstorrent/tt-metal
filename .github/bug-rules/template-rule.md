# Template Rule — [Bug Pattern Name]

## Description

Describe what the bug is, why it happens, and the conditions under which it manifests.
Include references to specific subsystems, data structures, or APIs involved.

Explain the root cause: is it a type mismatch, an off-by-one, a stale value, a missing
validation, etc.?

## What to Look For

1. **Pattern 1**: Describe a specific code pattern that indicates this bug.

2. **Pattern 2**: Describe another variant of the same bug class.

3. **Pattern 3**: Additional patterns as needed.

## Bad Code Examples

```cpp
// BUG: Explain why this code is buggy
auto x = some_function(wrong_argument);
```

## Good Code Examples

```cpp
// GOOD: Explain why this code is correct
auto x = some_function(correct_argument);
```
