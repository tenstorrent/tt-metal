CreateDataMovementKernel
=========================

.. doxygenfunction:: CreateDataMovementKernel(Program &program, const std::string &file_name, const CoreCoord &core, const std::vector<uint32_t> &compile_args, DataMovementProcessor processor_type, NOC noc);

.. doxygenfunction:: CreateDataMovementKernel( Program &program, const std::string &file_name, const CoreCoord &core, DataMovementProcessor processor_type, NOC noc);

.. doxygenfunction:: CreateDataMovementKernel( Program &program, const std::string &file_name, const CoreRange &core_range, const std::vector<uint32_t> &compile_args, DataMovementProcessor processor_type, NOC noc);

.. doxygenfunction:: CreateDataMovementKernel( Program &program, const std::string &file_name, const CoreRange &core_range, DataMovementProcessor processor_type, NOC noc);

.. doxygenfunction:: CreateDataMovementKernel( Program &program, const std::string &file_name, const CoreRangeSet &core_ranges, const std::vector<uint32_t> &compile_args, DataMovementProcessor processor_type, NOC noc);

.. doxygenfunction:: CreateDataMovementKernel( Program &program, const std::string &file_name, const CoreRangeSet &core_ranges, DataMovementProcessor processor_type, NOC noc);
