.. _Environment Variables:

Environment Variables
=====================

TT_METAL_HOME
-------------

**Optional:** Needed only for specific workflows.

**Description:**

``TT_METAL_HOME`` is an environment variable that points to the root directory of the tt-metal repository. It may be needed before running some tests, or scripts. It is not mandated by the metal runtime.

**Usage:**

.. code-block:: bash

   export TT_METAL_HOME=/path/to/your/tt-metal

**When is it needed:**

- **Development:** Used by test scripts and build tools to locate repository files
- **Documentation builds:** Referenced when checking spelling and building docs
- **CMake builds:** Used as a fallback when building tt-train standalone

TT_METAL_RUNTIME_ROOT
---------------------

**Required:** Conditional (only for specific C++ application scenarios)

**Description:**

``TT_METAL_RUNTIME_ROOT`` serves as an override to tell ``libtt_metal.so`` where to look for artifacts - firmware blobs, etc.. that is needed during runtime.

**When is it required:**

This variable is **only** required when running a C++ application (not ttnn Python) **AND**:

- Not using the metalium prebuilt binary package
- Not running from a repository clone directory

**Usage:**

Set this variable to point to the directory containing runtime artifacts:

.. code-block:: bash

   export TT_METAL_RUNTIME_ROOT=/path/to/runtime/artifacts

**Background:**

When running Python applications with ttnn, the runtime root is automatically determined by the package installation location. However, C++ applications that link against ``libtt_metal.so`` need to know where to find runtime files like kernel binaries and device firmware.

The library uses the following fallback order to locate runtime artifacts:

1. ``TT_METAL_RUNTIME_ROOT`` environment variable (if set)
2. Installation directory (for prebuilt packages)
3. Repository root (for development builds)

**Example scenarios:**

- **Scenario 1 (No override needed):** Running from repo clone

  .. code-block:: bash

     cd ~/tt-metal
     ./build/my_cpp_app  # Uses repo root automatically

- **Scenario 2 (Override required):** Custom C++ application outside repo

  .. code-block:: bash

     export TT_METAL_RUNTIME_ROOT=/opt/tt-metalium-runtime
     /usr/local/bin/my_cpp_app

TT_METAL_JIT_SCRATCH
--------------------

**Optional:** Recommended for multi-host MPI environments

**Description:**

``TT_METAL_JIT_SCRATCH`` specifies a local (non-NFS) directory where the JIT build system writes temporary compilation artifacts during the SFPI compilation phase. This avoids NFS contention and ESTALE (stale file handle) errors that can occur when multiple MPI ranks simultaneously write to a shared NFS cache directory.

**How it works:**

The JIT build system uses a hybrid map-reduce approach:

1. **Compile Phase:** Kernel source files are compiled to the scratch directory on local disk
2. **Merge Phase:** Successfully compiled artifacts are atomically copied/hardlinked to the shared NFS cache

This separation ensures that the I/O-intensive compilation phase doesn't contend with other hosts for NFS bandwidth, while still maintaining a shared cache for reuse across runs.

**Usage:**

.. code-block:: bash

   export TT_METAL_JIT_SCRATCH=/tmp/tt-jit-build

**When to use it:**

- **Multi-host MPI jobs:** Essential when running distributed workloads across multiple hosts that share an NFS filesystem
- **High-concurrency scenarios:** Recommended when many processes (>10) may compile kernels simultaneously
- **NFS stability issues:** Use if you encounter ESTALE errors or slow compilation due to NFS latency

**Default behavior:**

If not set, the scratch directory defaults to a rank-scoped subdirectory under ``/tmp/tt-jit-build`` (e.g., ``/tmp/tt-jit-build/<hostname>_rank_0/``) when using ``tt-run`` with MPI.

**Integration with tt-run:**

When using ``tt-run`` for distributed execution:

- If ``TT_METAL_JIT_SCRATCH`` is inherited from the launcher environment (or unset), ``tt-run`` appends a rank-specific suffix (``<hostname>_rank_<N>``).
- If ``TT_METAL_JIT_SCRATCH`` is explicitly set in rank-binding YAML (``global_env`` or ``env_overrides``), ``tt-run`` preserves that exact value and does not append a suffix.

Related behavior:

- ``TT_METAL_CACHE`` remains shared across ranks/hosts to maximize cache reuse.
- ``TT_METAL_LOGS_PATH`` is rank-scoped by ``tt-run`` unless explicitly overridden in rank-binding YAML.
