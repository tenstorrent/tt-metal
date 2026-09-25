.. _supported_data_types:

Supported Data Types
=====================

Reference for ``tt::tt_metal::DataType`` — the dtype a ``Tensor`` is created with, and what's actually usable per architecture.

.. list-table:: ``tt::tt_metal::DataType`` hardware compute support
    :widths: 15 12 12 12 49
    :header-rows: 1

    * - ``DataType``
      - Wormhole
      - Blackhole
      - Quasar
      - Notes
    * - ``BFLOAT16``
      - Yes
      - Yes
      - Yes
      -
    * - ``FLOAT32``
      - Yes
      - Yes
      - Yes
      -
    * - ``BFLOAT8_B``
      - Yes
      - Yes
      - No
      - Block-floating-point; see :ref:`ttnn.DataType` in the ttnn tensor docs for the block-float layout and tile-width caveats.
    * - ``BFLOAT4_B``
      - Yes
      - Yes
      - No
      - Block-floating-point, same caveats as ``BFLOAT8_B`` with coarser mantissa.
    * - ``UINT32``
      - Yes
      - Yes
      - No
      -
    * - ``UINT16``
      - Yes
      - Yes
      - No
      -
    * - ``UINT8``
      - Yes
      - Yes
      - Yes
      -
    * - ``INT32``
      - Yes
      - Yes
      - Yes
      -
    * - ``INT8``
      - Yes
      - Yes
      - Yes
      -
    * - ``FP8_E4M3``
      - No
      - Yes
      - Yes
      -  Hardware/enum-legal on Blackhole and Quasar, but as of this writing the ``DataType`` is only wired up for one op path (the DeepSeek V3 prefill combine/dispatch ops), row-major layout only — check op support before opting in rather than assuming general elementwise support.
    * - ``INVALID``
      - —
      - —
      - —
      - Not a representable tensor value. Used as a sentinel default in op-attribute structs (e.g. ``output_dtype == DataType::INVALID`` meaning "inherit the input tensor's dtype") — see ``tt::tt_metal::datatype_to_dataformat_converter``, which throws if called with any value other than the ones in this table.

.. note::

   "Supported" above means the Tensix compute engine supports the underlying ``DataFormat`` on that architecture (via ``tt::is_data_format_supported``). It does **not** mean every ttnn op has implemented that dtype end-to-end (padding, tilize/untilize, etc.) — ``FP8_E4M3`` above is a concrete example of that gap. Op-by-op dtype coverage changes too often to track accurately here.

New ``DataType`` values are appended after ``INVALID`` rather than inserted in logical order, to keep previously-serialized tensor values stable — see the numbering of ``INVALID = 10`` followed by later additions in ``tensor_types.hpp``.

Background: DataType vs. DataFormat
-------------------------------------

Metalium has two distinct type systems, and "is this data type supported" depends on which one you mean:

* ``tt::tt_metal::DataType`` (``tt_metal/api/tt-metalium/tensor/tensor_types.hpp``) — the table above. The type most host-side/ttnn code interacts with.
* ``tt::DataFormat`` (``tt_metal/api/tt-metalium/tt_backend_api_types.hpp``) — the type a circular buffer, kernel unpacker/packer, or the Tensix compute engine operates on directly. Its own doc comment describes it as "the union of all data formats supported by Tensix hardware of all generations," with per-architecture legality checked at runtime, not by the enum itself.

Every ``DataType`` maps to exactly one ``DataFormat``, via ``tt::tt_metal::datatype_to_dataformat_converter``:

.. list-table:: ``DataType`` → ``DataFormat`` mapping
    :widths: 30 30
    :header-rows: 1

    * - ``DataType``
      - ``DataFormat``
    * - ``BFLOAT16``
      - ``Float16_b``
    * - ``FLOAT32``
      - ``Float32``
    * - ``BFLOAT8_B``
      - ``Bfp8_b``
    * - ``BFLOAT4_B``
      - ``Bfp4_b``
    * - ``UINT32``
      - ``UInt32``
    * - ``UINT16``
      - ``UInt16``
    * - ``UINT8``
      - ``UInt8``
    * - ``INT32``
      - ``Int32``
    * - ``INT8``
      - ``Int8``
    * - ``FP8_E4M3``
      - ``Fp8_e4m3``
    * - ``INVALID``
      - *(none — throws if converted)*

The reverse is not true: most ``DataFormat`` values have no corresponding ``DataType`` and can only be reached by working directly with circular buffers and kernels, below the Tensor abstraction. See :ref:`Runtime Data Formats <runtime_data_formats>` below for the full picture.

.. note::

   ``Buffer`` and ``MeshBuffer`` (the device-resident and multi-device memory allocations underneath a Tensor) are untyped — they carry no ``DataType`` or ``DataFormat`` at all, just raw bytes at a given size. Type only enters the picture once a ``Tensor`` is layered on top, or once a kernel's circular buffer is configured with a ``DataFormat``.

.. _runtime_data_formats:

Runtime Data Formats
---------------------

``tt::DataFormat`` (``tt_metal/api/tt-metalium/tt_backend_api_types.hpp``) is the complete, lower-level enum used by circular buffers and kernels. Compute-engine legality per architecture is decided by ``tt::is_data_format_supported(format, arch)``; not every value in the enum represents an "input/output element type" in the same sense — some exist purely to move bytes without the compute engine interpreting them, and a few are declared but not yet wired to any architecture.

.. list-table:: ``tt::DataFormat`` compute-engine support and role
    :widths: 16 12 12 12 48
    :header-rows: 1

    * - ``DataFormat``
      - Wormhole
      - Blackhole
      - Quasar
      - Role
    * - ``Float16_b``, ``Float32``, ``Int8``, ``Int32``, ``UInt8``
      - Yes
      - Yes
      - Yes
      - Element type, reachable from a Tensor ``DataType`` (see table above).
    * - ``Bfp8_b``, ``Bfp4_b``, ``UInt16``, ``UInt32``
      - Yes
      - Yes
      - No
      - Element type, reachable from a Tensor ``DataType``.
    * - ``Fp8_e4m3``
      - No
      - Yes
      - Yes
      - Element type, reachable from a Tensor ``DataType`` (narrowly, see caveat above).
    * - ``Float16``, ``Lf8``, ``Tf32``
      - Yes
      - Yes
      - Yes
      - Element type at the kernel/circular-buffer level only — no ``DataType`` maps to these. ``Tf32`` is additionally declared but throws ``"unsupported atm"`` from ``tile_size()``/``datum_size()``; treat it as reserved, not usable.
    * - ``Bfp2``, ``Bfp2_b``, ``Bfp4``, ``Bfp8``
      - Yes
      - Yes
      - No
      - Block-floating-point element type at the kernel level only. (An in-flight, unmerged change exposes ``Bfp2_b`` as a Tensor ``DataType`` once a consuming op — matmul — supports it; not yet in this branch.)
    * - ``Int16``
      - No
      - No
      - Yes
      - Element type at the kernel level only, Quasar-specific.
    * - ``MxFp4``, ``MxFp6P``, ``MxFp6R``, ``MxFp8P``, ``MxFp8R``, ``MxInt2``, ``MxInt4``, ``MxInt8``
      - No
      - No
      - Yes
      - Microscaling block formats, Quasar-only, kernel level only — no ``DataType`` exposes these yet.
    * - ``RawUInt8``, ``RawUInt16``, ``RawUInt32``
      - Yes
      - Yes
      - Yes
      - Not an element type. Internal pack/unpack passthrough formats used by the kernel build (``tt_metal/jit_build/data_format.cpp``) to move bytes without the compute engine reinterpreting them (e.g. ``RawUInt16`` is substituted for ``Float16`` during unpack-format selection).
    * - ``MxFp4_2x_A``, ``MxFp4_2x_B``
      - No
      - No
      - No
      - Declared in the enum but not wired into ``is_data_format_supported`` for any architecture as of this writing. Treat as reserved/placeholder.
    * - ``Invalid``
      - —
      - —
      - —
      - Sentinel only, not a data format.
