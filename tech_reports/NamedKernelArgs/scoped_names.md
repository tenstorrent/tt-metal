# Scoped names: composing device code into one kernel

*Metal 2.0. Ordinary kernels don't need this.*

Scoped names are for a kernel that combines several pieces of device code, or several uses of one piece, each with its own arguments and bindings. Kernel fusion works this way.

- **Host:** write any binding `accessor_name`, argument name, or tensor binding `sequence_name` as `"<scope>.<name>"`.
- **Kernel:** reach it as `<scope>::<name>`. All of a kernel's names in one scope, of every kind, are static members of one generated struct, named after the scope.
- A scope is a C++ type, so device code can be written once, as a template, and used with each scope.

Unscoped names work as they always have: `args::<name>`, `dfb::<name>`, and so on.

## Example

Host:

```cpp
kernel.runtime_arg_schema.runtime_arg_names = {"a.num_tiles", "b.num_tiles"};
kernel.compile_time_args = {{"a.Ht", 8}, {"b.Ht", 16}};
kernel.dfb_bindings = {ConsumerOf(DFBSpecName{"x"}, "a.in"), ConsumerOf(DFBSpecName{"y"}, "b.in")};

// Run args use the full name.
kernel_run_args.runtime_arg_values["a.num_tiles"][node] = 4;
```

Kernel:

```cpp
template <typename Scope>
void op() {
    constexpr uint32_t ht = get_arg(Scope::Ht);
    uint32_t num_tiles = get_arg(Scope::num_tiles);
    DataflowBuffer in(Scope::in);
    // ...
}

void kernel_main() {
    op<a>();
    op<b>();
}
```

The JIT build generates `kernel_scopes_generated.h`, which is included before the kernel source:

```cpp
struct a {
    static constexpr ::experimental::RtaArg<uint32_t> num_tiles{0};
    static constexpr ::experimental::CtaVal<uint32_t> Ht{8u};
    static constexpr ::DFBBindingToken<...> in{...};
    // ...plus the by-name lookups below
};
struct b { /* ... */ };
```

Scoping changes only where a name is emitted. Dispatch buffer layouts and the kernel hash work as they do for unscoped names, and run args are keyed by the full name.

## By-name lookups

Each scope gets its own counterparts of `dfb::get_token_if_present`, `tensor::get_token_if_present` and `scratch::get_token_if_present`:

```cpp
a::get_dfb_token_if_present<"in">()                    // likewise get_tensor_token_if_present, get_scratch_token_if_present
Scope::template get_dfb_token_if_present<"in">()       // through a template parameter
```

The unscoped `dfb::get_token_if_present<"in">()` doesn't see scoped bindings: it returns `nullptr` for `a.in`.

## Rules

Validation enforces all of these, with an error that names the rule.

- There is one level of scope (`"a.x"`, not `"a.b.x"`). The scope and the name are each a C++ identifier.
- Names in one scope are unique across all kinds: an argument and a DFB binding can't share a name. They also mustn't collide with the scope's own name (`"op.op"`), or with the members the generator adds: the by-name lookups, and the type alias `<name>_t` of each tensor binding.
- A scope can't be named after a namespace that the generated headers declare or use: `args`, `dfb`, `pipe`, `sem`, `scratch`, `tensor`, `sem_internal`, `binding_details`, `experimental`, `internal`, `std` or `tensor_accessor`. A clash with any other global name in device code shows up as a kernel compile error.
- A scoped tensor binding sequence's members must be in the sequence's own scope.
- The accessor-name length limit (`MAX_ACCESSOR_NAME_LENGTH`) applies to the whole `"<scope>.<name>"`.
- Scopes belong to their KernelSpec. The same scope name in two KernelSpecs relates nothing.

## Sharing one binding between scopes

Bind the resource once, under one scope's name; for most binding kinds, a second binding of the same resource in one kernel is an error. To make the binding visible in a second scope too, alias it in kernel code, in a view of that scope:

```cpp
// b's names, plus a's DFB binding under b's name `in`. It's the same token, not a copy.
struct b_view : b { static constexpr const auto& in = a::in; };

op<b_view>();
```

Each piece of code can construct its own device objects from its own names, as a standalone kernel would, with one exception for DFBs:

- **On Quasar data-movement kernels with implicit sync (the default there), construct one `DataflowBuffer` per DFB endpoint and share it by reference.** Each object keeps its own running count of the tiles it has moved, and waits on the hardware counters relative to that count. So a second object would count from zero: it could read entries the first object already consumed, and would return credits a second time. Each piece of code can still take the binding token from its own scope, for the compile-time information LLK calls need.
- **Elsewhere** (Wormhole, Blackhole, and compute kernels), the sync state lives in the shared DFB interface, so one `DataflowBuffer` per scope works. Two objects for one endpoint can confuse the bookkeeping of device-side debug tooling, though.

A DFB bound in two *different* roles, produced in one scope and consumed in another, needs no alias. That's two bindings, a self-loop, wherever self-loops are allowed.

## Not supported yet

- Scoped arguments of a `TT_KERNEL` entry. It takes its arguments as parameters, so they must be unscoped.
- Scoped names under emulation.

## Examples in the tests

- `tests/tt_metal/tt_metal/api/metal2_host_api/kernel_compilation_tests/scoped_names/`: every kind of scoped name, templated device code, by-name lookups, and the sharing pattern.
- `ProgramSpecHWTest.ScopedArgsReadback` in `tests/tt_metal/tt_metal/api/metal2_host_api/integration_tests/kernel_args_loopbacks.cpp`: scoped argument values on hardware.
- `tests/tt_metal/tt_metal/api/metal2_host_api/unit_tests/invariant_tests/kernel_spec/scoped_names.cpp`: the validation rules.
