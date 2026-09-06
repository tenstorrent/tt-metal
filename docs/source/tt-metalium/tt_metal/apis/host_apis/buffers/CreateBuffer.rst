CreateBuffer
=================

.. deprecated::
   All ``CreateBuffer`` overloads are deprecated in favor of ``tt::tt_metal::distributed::MeshBuffer::create``.
   They will be removed after 2026-10-04.

.. doxygenfunction:: tt::tt_metal::CreateBuffer(const BufferConfig &config);
.. doxygenfunction:: tt::tt_metal::CreateBuffer(const ShardedBufferConfig &config);
.. doxygenfunction:: tt::tt_metal::CreateBuffer(const BufferConfig &config, DeviceAddr address);
.. doxygenfunction:: tt::tt_metal::CreateBuffer(const ShardedBufferConfig &config, DeviceAddr address);
.. doxygenfunction:: tt::tt_metal::CreateBuffer(const BufferConfig &config, SubDeviceId sub_device_id);
.. doxygenfunction:: tt::tt_metal::CreateBuffer(const ShardedBufferConfig &config, SubDeviceId sub_device_id);
