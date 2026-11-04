# Device versions of observability/api headers

Device code includes `<observability/api/...>` from one folder that the JIT build stages in its cache,
`<cache>/include/<hash>/observability/api/` (`tt_metal/jit_build/staged_includes.hpp`). The folder is a copy of:

- every header of `tt_metal/observability/api/`, the host's public folder, and
- every header of this folder.

A header here replaces the host header of the same name, which is then not copied. Use it when the device needs a
definition the host does not: a device-only struct, or a device-side counterpart that differs from the host's. A
header here with a new name is device-only. A metadata type defined here with fields that differ from the host's has a
different signature, so the host's `Metadata::as<T>()` does not match its events.

Device code never includes from the host tree; it sees these headers only through the staged copy. The same include
rules apply here as in `tt_metal/observability/api/`: a header includes only `observability/api/` headers, gets
standard headers through `observability/api/base.h`, and has no conditional on the side being compiled. The
`validate-observability-api` pre-commit hook checks both folders.

A header added here must also be listed in `HW_JIT_API_HEADERS` (`tt_metal/hw/sources.cmake`), so that an installed
runtime has it to stage.
