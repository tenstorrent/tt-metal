// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <fstream>

#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

// ========================================================
//                      CLASSES
// ========================================================

using namespace std;

TensorData::TensorData(vector<int> shape) {
    this->shape = shape;
    this->rank = shape.size();
    this->volume = vector_product(shape);

    this->data.resize(this->volume);
    int* data_ptr = this->data.data();
    for(int64_t i=0; i < this->volume; i++) data_ptr[i] = i;
}

void TensorData::print() {
    bool DEBUG = true;
    if (DEBUG) tt::log_debug(tt::LogDTX, "Printing TensorData");

    vector<int> counter = zeros(this->rank);
    for (int i=0; i<this->volume; i++){
        if (DEBUG) tt::log_debug(tt::LogDTX, "i = {}.  counter = {}", i, v2s(counter));

        int index = 0;   // = y*this->shape[0] + x;
        for (int d=0; d<rank; d++) {
            index += counter[d] * shape [d];
        }


        // Incrementing counter
        counter.back()++;
        for (int d=rank-1; d>0; d--) {
            if (counter[d] == this->shape[d]) {
                counter[d-1]++;
                counter[d] = 0;
            }
        }
    }

    /*
     Please redo the logging in this block if you need again.
    for (int y=0; y<this->shape[0]; y++){
        for (int x=0; x<this->shape[0]; x++){
            int index = y*this->shape[0] + x;
            cout << this->data[index];
            if (x<this->shape[0]-1)
                cout << ",";
        }
        cout << endl;
    }
    cout << endl;
    */
}

void TensorData::generate_csv(string filename){
    // TODO - RK / NS: Why is this set to true every time?
    bool DEBUG = true;

    string full_filename;
    full_filename.append(filename);
    full_filename.append(".csv");
    ofstream myfile(full_filename);


    if (DEBUG) tt::log_info(tt::LogDTX, "Generating csv file: {}", full_filename);

    for (int y=0; y<this->shape[0]; y++){
        for (int x=0; x<this->shape[0]; x++){
            int index = y*this->shape[0] + x;
            myfile << this->data[index];
            if (x < this->shape[0]-1)
                myfile << ",";

        }
        myfile << endl;
    }
    myfile << endl;

    // Close the file
    myfile.close();

}


int DTXTensor::volume() {
    assert(this->rank > 0);

    int volume = 1;
    assert(this->end.size() == this->rank);
    assert(this->str.size() == this->rank);
    for (int d=0; d<this->rank; d++) {
        assert(this->end[d] >= this->str[d]);
        int dim_size = this->end[d] - this->str[d] + 1;
        volume = volume * dim_size;
    }
    return volume;
}

void DTXTensor::print() {
    tt::log_info(tt::LogDTX, "{}", this->get_string());
}

string DTXTensor::get_string() {
    string str;
    string end;
    str.append("[");
    end.append("[");

    for (int i=0; i<this->rank; i++) {
        str.append(to_string(this->str[i]));
        end.append(to_string(this->end[i]));

        if (i != this->rank-1) {
            str.append(",");
            end.append(",");
        }
    }
    str.append("]");
    end.append("]");

    string out;
    out.append(str);
    out.append("->");
    out.append(end);

    return out;
}

void TensorPair::print_string(){
    tt::log_info(tt::LogDTX, "{}", this->get_string());
}

string TensorPair::get_string() {
    string out;

    out.append("SRC: group=");
    out.append(to_string(this->src_group));
    out.append(", Tensor=");
    out.append(this->src_tensor->get_string());
    out.append(" ==>> DST: Tensor=");
    out.append(this->dst_tensor->get_string());
    return out;
}

string Transfer::get_string() {
    string out;
    out.append("SRC: soc_core=");
    out.append(v2s(this->src_soc_core));
    out.append(", address=");
    out.append(to_string(this->src_address));
    out.append("  ==>  DST: address=");
    out.append(to_string(this->dst_address));
    out.append(", size=");
    out.append(to_string(this->size));
    return out;
}

void TransformationNode::print(int spaces) {

    tt::log_debug(tt::LogDTX, "{}Transformation Node: opcode = {}", s(spaces), this->opcode);

    int group_index = 0;
    for (TensorPairGroup * group : this->groups) {

        tt::log_debug(tt::LogDTX, "{}Group = {}; shape = {}, core = {}", s(2 + spaces), group_index, v2s(group->shape), v2s(group->core));

        tt::log_debug(tt::LogDTX, "{}TensorPairs ({}):", s(4+spaces), group->tensor_pairs.size());
        int tp_index = 0;
        for (TensorPair * tp : group->tensor_pairs) {
            tt::log_debug(tt::LogDTX, "{}TensorPair[{}]{}", s(6+spaces), tp_index, tp->get_string());
            tp_index++;
        }

        tt::log_debug(tt::LogDTX, "{}Transactions:", s(4+spaces));
        int tx_index = 0;
        for (Transfer * tx : group->transfers) {
            tt::log_debug(tt::LogDTX, "{}, Transaction[{}]  {}", s(6+spaces), tx_index, tx->get_string());
            tx_index++;
        }
        group_index++;
    }
}

void DataTransformations::print() {
    this->print(0);
}

void DataTransformations::print(int spaces) {
    tt::log_debug(tt::LogDTX, "{} DataTransformations -- nodes = {}", s(spaces), this->transformations.size());
    tt::log_debug(tt::LogDTX, "{} ----------------------------------------------------", s(spaces));
    for (int t=0; t<this->transformations.size(); t++) {
        this->transformations[t]->print(spaces+3);
    }
}

bool DataTransformations::compare_to_golden(TransformationNode * golden) {
    bool pass = true;
    return true;
}
