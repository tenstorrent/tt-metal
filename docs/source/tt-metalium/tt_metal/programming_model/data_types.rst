.. _supported_data_types:

Supported Data Types
=====================

If you're bringing data into Metalium and asking "can I even use this dtype," this page answers that directly — no need to understand Metalium's internal type system first.

.. list-table:: What data can a ``Tensor`` hold?
    :widths: 16 12 20 52
    :header-rows: 1

    * - Data type
      - Supported?
      - Architectures
      - Notes
    * - float32
      - Yes
      - Wormhole, Blackhole, Quasar
      - IEEE 754 binary32.
    * - bfloat16
      - Yes
      - Wormhole, Blackhole, Quasar
      - Not IEEE 754 — same exponent range as float32, with a truncated mantissa.
    * - float16 (IEEE half-precision)
      - No
      -
      - The hardware can operate on this at the kernel level, but no Tensor dtype exposes it. Cast to bfloat16 or float32.
    * - tf32
      - No
      -
      - The matrix engine natively supports this in hardware, but no Tensor dtype exposes it yet, and Metalium's own tile/datum-size plumbing isn't wired up for it either. Cast to bfloat16 or float32.
    * - block-float8 (8-bit block-floating-point)
      - Yes
      - Wormhole, Blackhole
      - A shared exponent covers a whole tile face rather than each element carrying its own — different tradeoffs than a per-element float. Not available on Quasar.
    * - block-float4
      - Yes
      - Wormhole, Blackhole
      - Same scheme as block-float8, coarser precision. Not available on Quasar.
    * - float8 (E4M3)
      - Narrowly
      - Blackhole, Quasar
      - Hardware-legal, but as of this writing only wired up for one specific op path (not general elementwise use) — check op support before relying on it.
    * - float8 (E5M2)
      - No
      -
      - Exists at the kernel level, but no Tensor dtype exposes it.
    * - int8
      - Yes
      - Wormhole, Blackhole, Quasar
      - Signed 8-bit, two's complement.
    * - uint8
      - Yes
      - Wormhole, Blackhole, Quasar
      - Unsigned 8-bit.
    * - int16
      - No
      -
      - No Tensor dtype exposes this; the closest hardware format is Quasar-only and kernel-level only. Cast to int32.
    * - uint16
      - Yes
      - Wormhole, Blackhole
      - Not available on Quasar.
    * - int32
      - Yes
      - Wormhole, Blackhole, Quasar
      - Signed 32-bit, two's complement.
    * - uint32
      - Yes
      - Wormhole, Blackhole
      - Not available on Quasar.
    * - int64, uint64, float64
      - No
      -
      - No 64-bit type exists anywhere in Metalium's runtime, at any level. Cast to the nearest 32-bit type.

.. note::

   "Supported" here means the Tensix compute engine can operate on the type on that architecture, and it's exposed as a Tensor dtype you can actually create. It does **not** guarantee every ttnn op has implemented that dtype end-to-end — float8 (E4M3) above is a concrete example of that gap. Op-by-op dtype coverage changes too often to track accurately on this page.

Working below the Tensor level
--------------------------------

Everything above is about ``Tensor``. If you're writing kernels or configuring circular buffers directly, Metalium exposes a much larger set of low-level formats than the table above — including formats no ``Tensor`` dtype can reach at all. That detail lives below; skip it unless you need it.

``Buffer`` and ``MeshBuffer`` (the device-resident and multi-device memory allocations underneath a Tensor) are untyped — they carry no type information at all, just raw bytes at a given size. Type only enters the picture once a ``Tensor`` is layered on top, or once a kernel's circular buffer is configured with a format.

Internally, Metalium has two distinct type enums:

* ``tt::tt_metal::DataType`` (``tt_metal/api/tt-metalium/tensor/tensor_types.hpp``) — the type a ``Tensor`` is created with. This is what the table above documents.
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

The reverse is not true: most ``DataFormat`` values have no corresponding ``DataType`` and can only be reached by working directly with circular buffers and kernels, below the Tensor abstraction — see the full ``DataFormat`` table below.

New ``DataType`` values are appended at the end of the enum rather than inserted in logical order, to keep previously-serialized tensor values stable.

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
      - Element type, reachable from a Tensor ``DataType`` (see mapping above).
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
