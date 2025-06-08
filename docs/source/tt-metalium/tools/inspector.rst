Inspector
=========

Overview
--------

The Inspector is a tool that provides insights into Metal host runtime. It is designed to be on by default and to have
minimal impact on performance.
It consists of two components: one that logs necessary data to do investigation and one that allows clients to connect
and query Metal host runtime data.

Enabling
--------

Configure the Inspector by setting the following environment variables:

.. code-block::

   export TT_METAL_INSPECTOR=1                              # optional: enable/disable the Inspector. Default is `1` (enabled).
   export TT_METAL_INSPECTOR_LOG_PATH=logging_path          # optional: set logging path. Default is `$TT_METAL_HOME/generated/inspector`
   export TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0  # optional: enable/disable stopping execution if the Inspector is not initialized properly. Default is `1` (enabled).
   export TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS=0     # optional: enable/disable warnings on logging write exceptions (like disk out of space). Default is `1` (enabled).

Enabling the Inspector will override `TT_METAL_RISCV_DEBUG_INFO` and debugging info will be generated for riscv elfs.
