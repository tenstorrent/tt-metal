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

TT_METAL_CACHE
--------------

**Optional:** Defaults to a per-user cache directory.

**Description:**

``TT_METAL_CACHE`` chooses where compiled kernels are cached on disk. A directory named
``tt-metal-cache`` is created under the path you give. When the variable is unset the runtime
uses ``$HOME/.cache/tt-metal-cache/``, falling back to ``/tmp/tt-metal-cache-<uid>/`` when
there is no home directory -- per-uid because ``/tmp`` is shared.

Inside the root, each build configuration gets a directory named after a build key::

   <cache root>/<build key>/firmware/
   <cache root>/<build key>/kernels/

**Usage:**

.. code-block:: bash

   export TT_METAL_CACHE=/scratch/$USER

TT_METAL_CACHE_MAX_SIZE
-----------------------

**Optional:** No bound by default, so nothing is evicted.

**Description:**

Upper bound on the kernel cache. When it is exceeded, whole build key directories are evicted
least recently used first until it fits. Setting this is what turns eviction on: a process that
deletes files another process may need should be something a machine's owner enables
deliberately.

Accepts a byte count with an optional binary suffix: ``50G``, ``512M``, ``1024K``, ``2T``. ``0``
means no bound. A value that cannot be parsed is reported and ignored.

Eviction runs in the background at startup, never on the compile path, and at most once an hour
per root. A directory in use by a running process is never evicted: each live process holds a
lock on the directory it is building into, and the operating system releases that lock even if
the process is killed outright. Directories written by builds that predate this mechanism carry
no such lock, cannot be distinguished from live ones, and are therefore never touched -- they
become eligible once a current build reuses them.

If a single directory is larger than the bound, it is kept rather than evicted: the bound cannot
be satisfied, so evicting would only force a rebuild and leave you in the same place.

**Usage:**

.. code-block:: bash

   export TT_METAL_CACHE_MAX_SIZE=20G
