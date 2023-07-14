#pragma once

#include <libs/tensor/tensor.hpp>
#include "tt_dnn/op_library/operation.hpp"

namespace tt {

namespace tt_metal {

#ifdef DEBUG

namespace operation_history {

inline std::string DEFAULT_FILE_PATH{"build/operation_history.csv"};

struct TensorRecord {
    const StorageType storage_type;
    const Shape shape;
    const DataType data_type;
    const Layout layout;
    const std::optional<MemoryConfig> memory_config;
    tt::stl::reflection::Attributes attributes() const;
};

struct OperationRecord {
    const std::string opcode;
    const tt::stl::reflection::Attributes attributes;
    const std::vector<TensorRecord> input_tensor_records;
};

namespace detail {

struct OperationHistory {

    ~OperationHistory();

    void append(OperationRecord&& record);
    void to_csv(const std::filesystem::path& file_path);

  private:
    std::vector<OperationRecord> records;
};

inline OperationHistory OPERATION_HISTORY{};

}

template<typename ... Args>
inline void append(Args&& ... args) {
    detail::OPERATION_HISTORY.append(std::forward<Args>(args)...);
}

}  // namespace operation_history

#endif

}  // namespace tt_metal

}  // namespace tt
