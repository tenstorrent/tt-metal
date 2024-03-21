EnqueueReadBuffer
==================

.. doxygenfunction:: EnqueueReadBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, std::vector<uint32_t>& dst, bool blocking)
.. doxygenfunction:: EnqueueReadBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, void * dst, bool blocking)
