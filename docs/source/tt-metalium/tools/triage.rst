tt-triage
=========

Overview
--------

tt-triage is a tool that provides insights into Metal execution. It is designed to be easily extensible.
It consists of data provider scripts that gather data from various sources (such as Inspector, tt-exalens, etc.) and
analysis scripts that process data supplied by the data provider scripts.
Since most of the scripts use Inspector data, before using tt-triage, ensure that you have enabled Inspector in your Metal host runtime.

Running
-------

To run tt-triage, execute the ``tools/tt-triage.py`` script using Python 3.10 or newer.

Example use with tt-metal:

.. code-block:: bash

    export TT_METAL_HOME=~/work/tt-metal
    ./build_metal.sh --build-programming-examples
    build/programming_examples/matmul_multi_core
    triage


Arguments
---------

You can run ``tt-triage --help`` to see the available options.
Some notable options include:

 - ``--initialize-with-noc1``: Initializes the debugger context with NOC1 enabled (default: False). Useful if NOC0 is not functioning.
 - ``--dev=<device_id>``: Specifies the device ID to analyze (default: ``in_use``). You can provide multiple ``--dev`` options to analyze several devices. Use ``all`` to analyze every device.
 - ``--remote-exalens``: Connects to a remote exalens server (default: False). This is helpful if tt-triage is fails during UMD initialization. If you encounter errors indicating UMD cannot be initialized, try this option. Open a new terminal and run ``tt-exalens --server`` before starting your workload. Then, in your original terminal, run ``tt-triage --remote-exalens``.

Multihost, MPI, and Inspector alignment
---------------------------------------

When you run under MPI (``mpirun``, Slurm, etc.), **each rank** can have its own Metal process, devices, and Inspector RPC server. Logs and triage must agree on **which rank** they target.

Rank-scoped log directories
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``TT_METAL_LOGS_PATH`` points at a shared parent (for example on NFS), the runtime may write per-rank data under:

.. code-block:: text

   <TT_METAL_LOGS_PATH>/<hostname>_rank_<N>/generated/inspector

**tt-triage** and **parse_inspector_logs** look for that layout when resolving the default inspector log directory. They pick the subdirectory that matches the process rank when one of these is set (first match wins):

- ``OMPI_COMM_WORLD_RANK`` (Open MPI)
- ``PMI_RANK`` (MPICH / Hydra / Slurm PMI)
- ``SLURM_PROCID`` (Slurm without PMI)
- ``PMIX_RANK`` (OpenPMIx / PRRTE)
- ``TT_MESH_HOST_RANK`` (Tenstorrent mesh; may duplicate across world ranks—prefer the standard MPI variables when both are set)

If no rank is detected, triage falls back to a ``*_rank_0`` directory when present, or the first matching path (with a warning if several exist). Use ``--inspector-log-path`` to select a directory explicitly when ambiguous.

Inspector RPC port (per rank)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Inspector gRPC server port is **base port + world rank** when a rank is detected from the environment (same precedence as above). The default base is **50051**, so rank 3 uses **50054** unless you override ``TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS`` / ``--inspector-rpc-port``.

.. note::

   **Port overflow**: ``base_port + max_rank`` must not exceed 65535. With the default base of 50051, up to ~15,000 ranks are safe. For larger jobs, lower the base port: set ``TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS=5000`` (or pass ``--inspector-rpc-port 5000``) to leave sufficient headroom.

When using **tt-run** (``ttnn.distributed.ttrun``), the default ``--inspector-rpc-port`` **50051** is adjusted the same way so timeout-launched triage on a non-zero rank connects to that rank’s Inspector.

Environment variables (summary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Role in multihost / triage
   * - ``TT_METAL_LOGS_PATH``
     - Root for generated logs; may contain ``<host>_rank_<N>`` children for MPI.
   * - ``TT_METAL_JIT_SCRATCH``
     - Local scratch for JIT builds; tt-run rank-scopes this path to reduce NFS contention.
   * - ``TT_METAL_CACHE``
     - Shared artifact cache (typically NFS); intentionally **not** rank-scoped when using tt-run defaults.
   * - ``TT_METAL_INSPECTOR``, ``TT_METAL_INSPECTOR_RPC``
     - Enable Inspector and RPC for live tt-triage attachment.
   * - ``TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS``
     - ``host:port`` base; effective port adds MPI rank as above.
   * - ``OMPI_COMM_WORLD_RANK`` / ``PMI_RANK`` / etc.
     - Used for rank-aware log paths and Inspector port selection in both C++ and triage Python.

For launching multi-rank jobs with controlled environment propagation, see **tt-run** in ``ttnn/ttnn/distributed/ttrun.py`` (help: ``tt-run --help``).

Extending
---------

Take a look at the ``tools/triage/tt-triage.md`` file for instructions on how to extend tt-triage with new data providers and analysis scripts.
