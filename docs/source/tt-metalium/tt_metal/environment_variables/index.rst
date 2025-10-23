.. _Environment Variables:

Environment Variables
=====================

TT_METAL_HOME
-------------

**Required:** Conditional (only when required by a given workflow)

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

``TT_METAL_RUNTIME_ROOT`` serves as an override to tell ``libtt_metal.so`` where to look for its runtime artifacts (files that get read from disk).

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
