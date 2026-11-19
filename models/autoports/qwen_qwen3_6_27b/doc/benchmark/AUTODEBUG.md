# AutoDebug: stale loaded extension after building `event_query`

## Finding

**High confidence: at initial inspection the rebuild had updated `build_Release/ttnn/_ttnn.so`, but the Python environment selected the older `ttnn/ttnn/_ttnn.so`.** That selected copy lacked the new binding name; current Python source unconditionally imports it. This explains the reported `ImportError` without a binding implementation defect or device operation. During final review, the selected extension was observed to have been refreshed; see the final snapshot below.

Inspection only: no TTNN import, native extension execution, build, install, copy, or device operations were performed. The only written artifact is this report. Findings describe the files inspected on 2026-11-17; a post-remedy import remains untested.

## Direct observations

1. `event_query_smoke.log:3–7` records an import failing in this checkout's `ttnn/ttnn/__init__.py:116`, with `cannot import name 'event_query' from 'ttnn._ttnn.events'`. The log reports initial configuration before that failure, consistent with the native extension already having initialized.
2. Using **`python_env/bin/python` (Python 3.12)**, static `importlib.machinery.PathFinder` lookups resolve:
   - `ttnn` → `/home/ttuser/qwen36-27b/tt-metal/ttnn/ttnn/__init__.py`
   - `ttnn._ttnn` → `/home/ttuser/qwen36-27b/tt-metal/ttnn/ttnn/_ttnn.so`

   These lookups did not import either module; `ttnn._ttnn` was absent from `sys.modules` afterward. The environment's `ttnn-custom.pth` adds this checkout's `ttnn` directory, and its editable finder also maps `ttnn` to `ttnn/ttnn`. The shell's unqualified `python` is a separate Python 3.10 installation, so it is not an equivalent diagnostic interpreter.
3. All three extension files are separate regular files, not symlinks. The initial byte inspection gave:

| Artifact, relative to checkout | Modification time (UTC) | Bytes | SHA-256 prefix | `event_query` binding literal / new docstring |
| --- | --- | ---: | --- | --- |
| `ttnn/ttnn/_ttnn.so` — selected by Python | 2026-11-16 23:22:51.033514 | 21,718,688 | `1f6b329e13af` | absent / absent |
| `build_Release/ttnn/_ttnn.so` — build output | 2026-11-17 02:48:07.973914 | 21,720,088 | `29d4c8834280` | present / present |
| `build_Release/lib/_ttnn.so` — staged installation | 2026-11-14 04:18:58 | 21,718,368 | `92e741541145` | absent / absent |

   The checks searched for `event_query\0` and the new docstring beginning `Return whether all mesh devices completed this event`. This is binary-content evidence of different binding generations, stronger than modification times alone; it is not a runtime attribute check. Installation rewrites RPATH, so installed and build-output hashes need not match even after a correct install.
4. Current source registers `event_query` unconditionally in `ttnn/cpp/ttnn-nanobind/events.cpp:74–79`, targeting `tt::tt_metal::distributed::EventQuery`. `ttnn/cpp/ttnn-nanobind/__init__.cpp:256,335` creates the `events` submodule and calls its registration function. `ttnn/ttnn/__init__.py:116–122` unconditionally imports the new name. There is no missing registration call in this source path. The native submodule also explains the traceback's `unknown location`; no separate `events.so` is expected.
5. The generated Ninja graph has `ttnn` as a phony dependency on `ttnn/_ttnn.so` (`build_Release/build.ninja:55491`). Its link edge has `POST_BUILD = :` at line 36084: no deployment copy follows the link. Installation is a separate step:
   - `ttnn/CMakeLists.txt:439–446`: component **`tt_pybinds`** installs the target into `${PROJECT_SOURCE_DIR}/ttnn/ttnn`.
   - `build_Release/ttnn/cmake_install.cmake:279–305`: the generated rule copies precisely `build_Release/ttnn/_ttnn.so` to the selected source-tree path and changes RPATH.
   - `ttnn/CMakeLists.txt:424–436`: `ttnn-runtime` / `tar` install to the library staging directory. The configured install prefix is this checkout's `build_Release`.

## Causal interpretation and alternatives

The observed sequence is explained by a build/deployment mismatch:

1. New Python source requires `event_query`.
2. `cmake --build build_Release --target ttnn` produces the new native extension but has no install/copy step.
3. Python selects the existing source-tree extension, which lacks the binding.
4. Its `events` submodule exists, but importing the new name fails at package initialization.

This accounts for the successful build, newer build artifact, older installed artifacts, and specific missing-name error. A typo or omitted registration is contradicted by current source and the new binary's literal/docstring. A device synchronization failure cannot explain this import-time missing attribute. `LD_LIBRARY_PATH=.../build_Release/lib` controls native dependency lookup; it does not replace the extension path selected by Python. Refreshing only `build_Release/lib/_ttnn.so` therefore does not repair the selected source-tree copy.

No additional source defect is established within this narrow investigation. Static inspection does not prove runtime registration of the new binary or native dependency ABI compatibility; those remain follow-up checks after deployment.

## Minimal remedy — proposed, not executed

From the checkout root, use the existing targeted install component:

```bash
cmake --install build_Release --component tt_pybinds
```

This deploys the already-built extension to the path Python selects and applies the configured install RPATH (`ttnn/CMakeLists.txt:325–326`). Here `build` points to `build_Release`, so the source copy's `$ORIGIN/../../build/lib` resolves to the existing staged dependency directory. No rebuild or pip reinstall is needed to refresh this binding. In particular, editable installation's `CMakeBuild.run()` returns before building/copying (`setup.py:212–215`), so rerunning editable pip installation is not the direct remedy.

For a checkout-local copy-only alternative:

```bash
cmake -E copy_if_different build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so
```

That retains the build RPATH rather than performing CMake's install rewrite. `readelf -d` shows both the currently selected copy and fresh build output already have the same absolute build-tree RUNPATH, so this is a viable local refresh; the component install is preferable. Neither remedy requires updating `build_Release/lib/_ttnn.so` to fix this selected extension, though a later general runtime/package installation should refresh staged artifacts too.

Validate in a **fresh process**, with the project interpreter, after applying the chosen remedy:

```bash
python_env/bin/python -c 'import ttnn; print(ttnn._ttnn.__file__); assert callable(ttnn.event_query)'
```

This proposed smoke checks import and attribute exposure without creating a device or calling `event_query`. It was not run during this investigation. Actual event-query semantics and device behavior are outside scope.

## Claim review

The first-draft diagnosis was checked against the source registration chain, static interpreter resolution, all three binary contents, Ninja link/deployment boundary, and generated component-install destination/RPATH rules. The headline is the initially stale selected artifact; timestamps alone, hypothetical binding bugs, and device behavior are not used as proof.

**Final snapshot:** the repeat fingerprint check detected that `ttnn/ttnn/_ttnn.so` had changed during the investigation, without an install/copy action by this investigator. It now has 21,720,088 bytes, SHA-256 `e93aaf27ef11cad5cc5ae090fbb09d814aaade13a331a99b7a3c5f1258a4fdb1`, and both the new binding literal and docstring. `readelf -d` now reports RUNPATH `$ORIGIN/build/lib:$ORIGIN/../../build/lib:$ORIGIN`. The build output and staged library retained their original fingerprints. This is consistent with the targeted component installation described above; the actor and command were not observed. The selected artifact is therefore no longer stale by this static check. Complete post-install import success remains unverified until the proposed smoke is run.
