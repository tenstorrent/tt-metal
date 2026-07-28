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
``tt-metal-cache`` is created under the path you give. When the variable is unset the
runtime uses ``$HOME/.cache/tt-metal-cache/``, falling back to
``/tmp/tt-metal-cache-<uid>/`` when there is no home directory. The trailing ``<uid>``
matters: ``/tmp`` is shared, and a common root would let one user's cleanup act on another
user's files.

Inside the root, each distinct build configuration gets its own directory named after a
build key::

   <cache root>/<build key>/firmware/
   <cache root>/<build key>/kernels/

Treat that layout as an implementation detail and use the ``tt-metal-cache`` command below
instead of removing files by hand.

**Usage:**

.. code-block:: bash

   export TT_METAL_CACHE=/scratch/$USER

TT_METAL_CACHE_MAX_SIZE
-----------------------

**Optional:** No size bound by default, so nothing is evicted automatically.

**Description:**

Upper bound on the size of the kernel cache. When the cache exceeds this size, whole build
key directories are evicted, least recently used first, until it fits again. Setting this is
what turns automatic eviction on: a process that deletes files another process may be using
should be something a machine's owner enables deliberately, so there is no size bound out of
the box. On a shared machine, ``20G`` to ``50G`` per user is a reasonable starting point.

Accepts a byte count with an optional binary suffix: ``50G``, ``512M``, ``1024K``, ``2T``.
``0`` means no bound. A value that cannot be parsed is reported and ignored.

Size is the only thing that triggers eviction. An entry is removed because the cache is over
budget, never merely because it is old: if you are under your bound, nothing is deleted. Use
``tt-metal-cache clear`` or a smaller ``--max-size`` when you want space back on demand.

Eviction happens in the background at startup, never on the compile path, and at most once
an hour per cache root. A directory in use by a running process is never evicted, so
trimming is safe while other jobs are compiling: each live process holds a lock on the
directory it is building into, and the operating system releases that lock even if the
process is killed outright.

Two things are deliberately never evicted automatically. A directory is only ever a candidate
if a build that participates in the locking protocol has claimed it, which it records by
creating an in-use marker inside the directory. Trees left by tt-metal builds that predate
this mechanism have no such marker, cannot be told apart from live ones, and are therefore
never touched; they become candidates as current builds reuse them, and
``tt-metal-cache prune-unmanaged`` removes whatever is left when you ask. Their space is
reported by ``tt-metal-cache stat`` but is not counted against the bound above. Separately, on
a filesystem where file locking cannot exclude other hosts, notably NFS mounted with
``local_lock`` or ``nolock``, automatic eviction refuses to run at all, because a lock held on
another host would be invisible.

**Usage:**

.. code-block:: bash

   export TT_METAL_CACHE_MAX_SIZE=20G

TT_METAL_CACHE_TRIM
-------------------

**Optional:** Trimming is enabled by default, though a size or age bound is still required
before anything is evicted.

**Description:**

Set to ``0`` to stop the runtime from ever trimming the kernel cache on its own. Useful for
continuous integration, and for anything that deliberately runs against an isolated cache
and wants full control over its contents. Accepts ``0``/``1``, ``true``/``false``,
``yes``/``no`` and ``on``/``off``; an unrecognised value is reported and the default kept.
Explicit ``tt-metal-cache`` invocations still work.

**Usage:**

.. code-block:: bash

   export TT_METAL_CACHE_TRIM=0

The tt-metal-cache command
--------------------------

``tt-metal-cache`` is the supported way to inspect and prune the kernel cache::

   tt-metal-cache stat             # size and per-entry usage, least recently used first
   tt-metal-cache trim             # evict until the cache is under a size bound
   tt-metal-cache clear            # evict everything not in use by a running process
   tt-metal-cache prune-unmanaged  # remove trees no current build has ever claimed

``--dry-run`` reports what a command would remove without changing anything, and ``--root``
and ``--max-size`` override the environment for one invocation. Pass ``--root`` the path the
runtime resolves, the one ending in ``tt-metal-cache``, not the parent directory you set
``TT_METAL_CACHE`` to.

Each user has their own cache root, so the command only ever touches your own files. If a
machine is out of disk because of another user's cache, you still need that user, or a
privileged cleanup, to reclaim it.
