// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/operation_history.hpp"

namespace tt {

namespace tt_metal {


#ifdef DEBUG

namespace operation_history {

namespace detail {

OperationHistory::~OperationHistory() {
    this->dump_to_csv();
}

void OperationHistory::append(OperationRecord&& record) {
    TT_ASSERT(record.input_tensor_records.size() <= 5);
    this->records.push_back(std::move(record));
}

constexpr auto column_names = []{
    return std::array{
        "Opcode",
        "Composite Parent Names",
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
    row.push_back(fmt::format("{}", record.composite_parent_names));
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

void OperationHistory::dump_to_csv() {
    std::ofstream output_file_stream(csv_file_name());
    write_header(output_file_stream);
    for (const auto& record : this->records) {
        write_record(output_file_stream, record);
    }
}

}  // namespace detail

const char* csv_file_name() {
    return std::getenv("OPERATION_HISTORY_CSV");
}

bool enabled() {
    return csv_file_name() != nullptr;
}

}  // namespace operation_history

#endif

}  // namespace tt_metal

}  // namespace tt
