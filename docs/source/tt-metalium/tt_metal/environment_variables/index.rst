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

**Optional:** Defaults to ``$HOME/.cache/tt-metal-cache/``, or ``/tmp/tt-metal-cache/`` when ``HOME`` is not set.

**Description:**

``TT_METAL_CACHE`` sets where compiled kernels and firmware are cached: the cache is the ``tt-metal-cache/`` directory inside it. Each build configuration gets its own ``<build_key>/`` directory in the cache, and every compile-time variant of a kernel gets its own directory below ``<build_key>/kernels/<kernel_name>/``. Several processes may share one cache.

**Usage:**

.. code-block:: bash

   export TT_METAL_CACHE=/path/to/cache   # kernels go to /path/to/cache/tt-metal-cache/

**Limiting the cache size:**

By default nothing is ever removed from the cache. The cap is ``max_size`` in ``tt-metal-cache.conf`` in the cache directory. The setting belongs to the cache, so every process that shares the cache uses the same cap. The easiest way to set it is ``TT_METAL_CACHE_MAX_SIZE``:

.. code-block:: bash

   export TT_METAL_CACHE_MAX_SIZE=20G

- If ``tt-metal-cache.conf`` does not exist yet, the first process that sees the variable writes ``max_size = 20G`` to it, logs that it did, and uses it. The file stays: later processes use it, with or without the variable, so setting the variable once caps the cache for every user and job that shares it.
- If the file exists, the file wins. A variable with a different value only logs a warning that names both values and the file, so shells with different settings cannot fight over one cache. To change the cap, edit or remove the file.
- If the file cannot be read or has no valid ``max_size``, a warning is logged and the variable is used.
- Without the variable, the file alone decides; no file means no cap.

The file can also be written by hand:

.. code-block:: bash

   # Default cache location; with TT_METAL_CACHE=/path/to/cache use /path/to/cache/tt-metal-cache/
   mkdir -p ~/.cache/tt-metal-cache
   echo "max_size = 20G" > ~/.cache/tt-metal-cache/tt-metal-cache.conf

Like ``ccache``, the cache keeps a running size counter in ``.stats`` in the cache directory. Each kernel compile and precompiled header build adds what it wrote, and when that takes the counter over ``max_size`` the cache is trimmed right away, also in the middle of a long run. At start-up a process only reads the counter. The whole cache is scanned only when the counter is missing or unreadable, when it is over the cap, or once a day, to correct the counter for files removed by hand or a process that died. A trim deletes the least recently used entries until the cache is at 80% of the cap, and logs one line with how much was removed. Entries are kernel variants and precompiled headers (``pch/<hash>/``, one per compiler flag set, about 28 MB each; they can be most of the cache).

- An entry counts as used when a process compiles or loads it, and a process that keeps using it marks it again every 7.5 minutes, so entries in use are not deleted however long the process runs. Without a cap nothing is marked.
- Entries used in the last 15 minutes are kept. The cache can stay above the cap while they are this recent.
- If another process deletes a kernel variant after this process's cache check, even while it is being loaded, the variant is compiled again, also with a remote compile server.
- Firmware and other files in the cache are counted but never deleted. If they alone are above 80% of the cap, nothing is deleted and a warning says the cap is below what cannot be deleted.
- Files that cannot be deleted (for example another user's) are reported in a warning and still count toward the size.
- Only one process trims a cache at a time; the others skip the check. The same lock guards the counter.

``max_size = 0`` turns the cap off, explicitly: nothing is tracked or kept up to date, so no ``.stats``, lock or trash file is written and nothing is marked or scanned. ``TT_METAL_CACHE_MAX_SIZE=0`` writes ``max_size = 0`` to a missing file like any other value, and a file with ``max_size = 0`` wins over a non-zero variable, with a warning.

``max_size`` uses ``ccache``'s size syntax: a number, optionally with a fraction (``1.5G``), and an optional unit ``k``, ``M``, ``G`` or ``T`` (powers of 1000) or ``Ki``, ``Mi``, ``Gi`` or ``Ti`` (powers of 1024), optionally followed by ``B``. Units are case-sensitive. A bare number means GiB, so ``max_size = 20`` is 20 GiB, not 20 bytes. ``0``, no file, or no ``max_size`` line means no cap. A cap below ``100M`` is taken as a typo: it is reported and ignored. The same syntax applies to ``TT_METAL_CACHE_MAX_SIZE``. Text after ``#`` is a comment. An invalid value is reported as a warning and skipped; when the file has several ``max_size`` lines, the last valid one wins.

Known limits:

- Files added to the cache by other means (firmware builds, kernels fetched from a remote compile server, files copied in by hand) are not added to the counter until the next scan, at most a day later.
- On NFS, marks use the server's clock and the 15-minute window uses the client's, so clocks more than 15 minutes apart defeat the window.
- A file hard-linked into several kernel variants is counted once, in whichever variant is scanned first, so deleting that variant may free less than counted.
- With an empty or relative ``TT_METAL_CACHE``, the cache is relative to the current directory.
