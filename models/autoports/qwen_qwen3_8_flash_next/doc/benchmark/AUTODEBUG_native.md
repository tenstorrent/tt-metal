# Native EventQuery binding link diagnosis

CPU-only source and ELF inspection; no target imports or device calls. The parent
observed import failure for `_benchmark_event_query.abi3.so` with undefined
`_ZN3fmt3v116detail11assert_failEPKciS3_` after importing TTNN.

The extension was built with `SPDLOG_FMT_EXTERNAL` but without the checkout's
`FMT_HEADER_ONLY=1`. Its inclusion of `tt-metalium/distributed.hpp` reaches fmt
through `mesh_coord.hpp` and `tt_stl/assert.hpp`. The resulting extension has
undefined dynamic references to fmt v11 `detail::assert_fail`, `detail::vformat_to`,
and `vformat`. `readelf -d` records only libtt_metal, C++/C runtime, and loader
dependencies: no libfmt dependency. The link command in
`tt/benchmark_native/build/CMakeFiles/_benchmark_event_query.dir/link.txt` links
only nanobind-static-abi3 and libtt_metal.

`nm -D build/lib/libtt_metal.so` confirms EventQuery is exported but the required
fmt assertion symbol is not. `nm -D build/lib/libfmt.so.11.1.4` confirms the missing
fmt functions are available in that library. The enclosing checkout build flags
in `build/build.ninja` use `FMT_HEADER_ONLY=1` together with
`SPDLOG_FMT_EXTERNAL`; the standalone extension omitted the former.

Supported minimal fix: add `FMT_HEADER_ONLY=1` to the extension's compile
definitions, matching TTNN/Metal's header usage, then rebuild. Linking the exact
checkout libfmt is an alternative but adds an unnecessary runtime dependency.
Do not rely on incidental globally loaded symbols from other extensions.

Discriminating validation after repair: `nm -D` should show no undefined fmt
symbols in the extension; retain its EventQuery reference resolved through
libtt_metal. The parent should then repeat the authorized server-environment
import and live MeshEvent readiness smoke. Build success alone does not prove
runtime nanobind type interoperability or hardware behavior.

`ldd -r` outside Python also reports undefined Python C API symbols, which are
normal for an extension loaded into a Python interpreter; those are distinct
from the unsupported fmt references. Both build/lib and build_Release/lib
libtt_metal paths resolve to the same checkout artifact.

The metadata fix is present independently: `PhaseRecorder.annotate` deep-copies
fields into ticket metadata, and the adapter invokes it for completed internal
prefill chunk spans before `end`.
