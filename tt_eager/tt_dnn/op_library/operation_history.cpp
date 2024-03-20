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

template <typename RowType>
void write_row(std::ofstream& output_file_stream, const std::size_t num_columns, const RowType& row) {
    TT_ASSERT(row.size() == num_columns, fmt::format("row.size()=={} and num_columns=={}",row.size(), num_columns));
    for (const auto& element : row) {
        output_file_stream << '"' << element << '"' << "\t";
    }
    output_file_stream << std::endl;
}



std::string create_shard_spec_buffers_str(const std::vector<ShardSpecBuffer>& shard_spec_buffers){
    std::stringstream ss;
    ss << "[";
    std::for_each(shard_spec_buffers.begin(), shard_spec_buffers.end(), [&ss](const auto &buffer) { ss << fmt::format("{}", buffer); ss << ";"; });
    ss << "]";
    return ss.str();
}

std::size_t write_header(
    std::ofstream& output_file_stream,
    const std::size_t num_attributes,
    const std::size_t num_input_tensors,
    const std::size_t num_dimensions) {
    auto column_names = std::vector<std::string>{"Opcode", "Composite Parent Names"};

    for (auto attribute_index = 0; attribute_index < num_attributes; attribute_index++) {
        column_names.push_back(fmt::format("Attribute {} Name", attribute_index));
        column_names.push_back(fmt::format("Attribute {} Value", attribute_index));
    }

    for (auto input_tensor_index = 0; input_tensor_index < num_input_tensors; input_tensor_index++) {
        column_names.push_back(fmt::format("Input Tensor {} Storage Type", input_tensor_index));
        for (auto dimension_index = 0; dimension_index < num_dimensions; dimension_index++) {
            column_names.push_back(fmt::format("Input Tensor {} Shape {}", input_tensor_index, dimension_index));
        }
        column_names.push_back(fmt::format("Input Tensor {} Data Type", input_tensor_index));
        column_names.push_back(fmt::format("Input Tensor {} Layout", input_tensor_index));
        column_names.push_back(fmt::format("Input Tensor {} Memory Config", input_tensor_index));
        column_names.push_back(fmt::format("Input Tensor {} Shard Spec", input_tensor_index));
    }

    write_row(output_file_stream, column_names.size(), column_names);
    return column_names.size();
}

void write_record(
    std::ofstream& output_file_stream,
    const std::size_t num_columns,
    const OperationRecord& record,
    const std::size_t num_attributes,
    const std::size_t num_input_tensors,
    const std::size_t num_dimensions) {
    std::vector<std::string> row;
    row.reserve(num_columns);

    row.push_back(record.opcode);
    row.push_back(fmt::format("{}", record.composite_parent_names));
    for (auto attribute_index = 0; attribute_index < num_attributes; attribute_index++) {
        if (attribute_index < record.attributes.size()) {
            const auto& [name, value] = record.attributes.at(attribute_index);
            row.push_back(fmt::format("{}", name));
            row.push_back(fmt::format("{}", value));
        } else {
            row.push_back("");
            row.push_back("");
        }
    }
    for (auto input_tensor_index = 0; input_tensor_index < num_input_tensors; input_tensor_index++) {
        if (input_tensor_index < record.input_tensor_records.size()) {
            const auto& tensor_record = record.input_tensor_records.at(input_tensor_index);
            row.push_back(fmt::format("{}", tensor_record.storage_type));
            for (auto dimension_index = 0; dimension_index < num_dimensions; dimension_index++) {
                if (dimension_index < tensor_record.shape.rank()) {
                    row.push_back(fmt::format("{}", tensor_record.shape[dimension_index]));
                } else {
                    row.push_back("");
                }
            }
            row.push_back(fmt::format("{}", tensor_record.data_type));
            row.push_back(fmt::format("{}", tensor_record.layout));
            row.push_back(fmt::format("{}", tensor_record.memory_config));
            row.push_back(fmt::format("{}", create_shard_spec_buffers_str(tensor_record.shard_spec_buffers)));
        } else {
            row.push_back("");
            for (auto dimension_index = 0; dimension_index < num_dimensions; dimension_index++) {
                row.push_back("");
            }
            row.push_back("");
            row.push_back("");
            row.push_back("");
            row.push_back("");
        }
    }
    write_row(output_file_stream, num_columns, row);
}

void OperationHistory::dump_to_csv() {
    if (not enabled())
        return;

    std::ofstream output_file_stream(csv_file_name());

    std::size_t num_attributes = 0;
    for (const auto& record : this->records) {
        num_attributes = std::max(num_attributes, record.attributes.size());
    }

    std::size_t num_input_tensors = 0;
    std::size_t num_dimensions = 0;
    for (const auto& record : this->records) {
        num_input_tensors = std::max(num_input_tensors, record.input_tensor_records.size());
        for (const auto& input_tensor_record : record.input_tensor_records) {
            num_dimensions = std::max(num_dimensions, input_tensor_record.shape.rank());
        }
    }

    auto num_columns = write_header(output_file_stream, num_attributes, num_input_tensors, num_dimensions);
    for (const auto& record : this->records) {
        write_record(output_file_stream, num_columns, record, num_attributes, num_input_tensors, num_dimensions);
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
