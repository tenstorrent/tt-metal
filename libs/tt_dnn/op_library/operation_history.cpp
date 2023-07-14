#include "tt_dnn/op_library/operation_history.hpp"

namespace tt {

namespace tt_metal {


#ifdef DEBUG

namespace operation_history {

tt::stl::reflection::Attributes TensorRecord::attributes() const {
    return {
        {"storage_type", fmt::format("{}", this->storage_type)},
        {"shape", fmt::format("{}", this->shape)},
        {"data_type", fmt::format("{}", this->data_type)},
        {"layout", fmt::format("{}", this->layout)},
        {"memory_config", fmt::format("{}", this->memory_config)},
    };
}

namespace detail {

OperationHistory::~OperationHistory() {
    this->to_csv(DEFAULT_FILE_PATH);
}

void OperationHistory::append(OperationRecord&& record) {
    TT_ASSERT(record.input_tensor_records.size() <= 5);
    this->records.push_back(std::move(record));
}

constexpr auto column_names = []{
    return std::array{
        "Opcode",
        "Attributes",
        "Input Tensor 0 Storage Type",
        "Input Tensor 0 Shape",
        "Input Tensor 0 Data Type",
        "Input Tensor 0 Layout",
        "Input Tensor 0 Memory Config",
        "Input Tensor 1 Storage Type",
        "Input Tensor 1 Shape",
        "Input Tensor 1 Data Type",
        "Input Tensor 1 Layout",
        "Input Tensor 1 Memory Config",
        "Input Tensor 2 Storage Type",
        "Input Tensor 2 Shape",
        "Input Tensor 2 Data Type",
        "Input Tensor 2 Layout",
        "Input Tensor 2 Memory Config",
        "Input Tensor 3 Storage Type",
        "Input Tensor 3 Shape",
        "Input Tensor 3 Data Type",
        "Input Tensor 3 Layout",
        "Input Tensor 4 Memory Config",
        "Input Tensor 4 Storage Type",
        "Input Tensor 4 Shape",
        "Input Tensor 4 Data Type",
        "Input Tensor 4 Layout",
        "Input Tensor 4 Memory Config",
    };
};

template<typename RowType>
void write_row(std::ofstream& output_file_stream, const RowType& row) {
    TT_ASSERT(row.size() == column_names().size());
    for (const auto& element : row) {
        output_file_stream << '"' << element << '"' << ",";
    }
    output_file_stream << std::endl;
}

void write_header(std::ofstream& output_file_stream) {
    write_row(output_file_stream, column_names());
}

void write_record(std::ofstream& output_file_stream, const OperationRecord& record) {
    std::vector<std::string> row;
    row.reserve(column_names().size());

    row.push_back(record.opcode);
    row.push_back(fmt::format("{}", record.attributes));
    for (const auto& tensor_record : record.input_tensor_records) {
        row.push_back(fmt::format("{}", tensor_record.storage_type));
        row.push_back(fmt::format("{}", tensor_record.shape));
        row.push_back(fmt::format("{}", tensor_record.data_type));
        row.push_back(fmt::format("{}", tensor_record.layout));
        row.push_back(fmt::format("{}", tensor_record.memory_config));
    }
    while (row.size() < column_names().size()) {
        row.push_back("");
    }
    write_row(output_file_stream, row);
}

void OperationHistory::to_csv(const std::filesystem::path& output_file_path) {
    std::ofstream output_file_stream(output_file_path);
    write_header(output_file_stream);
    for (const auto& record : this->records) {
        write_record(output_file_stream, record);
    }
}

}  // namespace detail

}  // namespace operation_history

#endif

}  // namespace tt_metal

}  // namespace tt
