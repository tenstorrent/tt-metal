// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

#include "tt_metal/common/logger.hpp"

// ========================================================
//                      CLASSES
// ========================================================

// skip copy-constructor and assign operators on these types
template <typename T>
struct tpl_nocopy {
        tpl_nocopy() = default;
        tpl_nocopy(const T& ) = delete;
        T operator=(const T& ) = delete;
};

class TensorData : tpl_nocopy<TensorData> {
    public:
        std::vector<int> data;
        int rank;
        std::vector<int> shape;
        int volume;

        explicit TensorData(std::vector<int> shape);

        void init_increasing();
        void print();
        void generate_csv(std::string filename);
};

class DTXTensor : tpl_nocopy<DTXTensor> {
    public:
        std::vector<int> str;
        std::vector<int> end;
        int rank = -1;

        DTXTensor(std::vector<int> str, std::vector<int> end){
            this->str = str;
            this->end = end;
            this->rank = this->str.size();
        }

        DTXTensor() {
            this->rank = 0;
        };

        int volume();

        void print();
        std::string get_string();
};

class TensorPair : tpl_nocopy<TensorPair> {
    public:
        DTXTensor * src_tensor;    // Tensor range
        int src_group;          // ID of the group to which the src_tensor is pointing
        DTXTensor * dst_tensor;    // Tensor Range

        TensorPair(DTXTensor * src_tensor,  DTXTensor * dst_tensor) {
            this->src_tensor = src_tensor;
            this->src_group = 0;
            this->dst_tensor = dst_tensor;
        }

        TensorPair(DTXTensor * src_tensor, int src_group, DTXTensor * dst_tensor) {
            this->src_tensor = src_tensor;
            this->src_group = src_group;
            this->dst_tensor = dst_tensor;
        }

        std::string get_string();
        std::string get_short_string();
        void print_string();
        ~TensorPair() {
            delete src_tensor;
            delete dst_tensor;
        }
};

class Transfer : tpl_nocopy<Transfer> {
    public:
    int src_address;
    int dst_address;
    int size;
    int pad;
    std::vector<int> src_soc_core;

    std::string get_string();

};

class TensorPairGroup : tpl_nocopy<TensorPairGroup> {
    public:
    std::vector<int> shape;
    std::vector<TensorPair *> tensor_pairs;

    std::vector<Transfer *> transfers;

    // Attributes
    int address;                // Address of buffer this is stored in (L1 or DRAM)
    std::vector<int> core = {-1,-1}; // Tensix core
    int streaming_id;           // The sequence order that's loaded into a CB or Buffer
    void delete_tensor_pairs() {
        for(auto tp : tensor_pairs) {
            delete tp;
        }
        tensor_pairs.clear();
    }
    void delete_transfers() {
        for(auto tr : transfers) {
            delete tr;
        }
        transfers.clear();
    }
    ~TensorPairGroup() {
        delete_tensor_pairs();
        delete_transfers();
    }
};

class TransformationNode : tpl_nocopy<TransformationNode> {
    public:

    std::string opcode;
    std::vector<TensorPairGroup *> groups;

    TransformationNode(std::string opcode, int number_of_groups) {
        this->opcode = opcode;
        for (int g=0; g<number_of_groups; g++){
            TensorPairGroup * group = new TensorPairGroup();
            this->groups.push_back(group);
        }
    }

    int create_new_group() {
        TensorPairGroup * new_group = new TensorPairGroup();
        this->groups.push_back(new_group);
        return this->groups.size()-1;
    }

    void print(int s);
    void delete_groups() {
        for(auto g : groups) {
            delete g;
        }
        groups.clear();
    }
    ~TransformationNode() {
        delete_groups();
    }
};



/*
class Buffer {
    public:

    int address;
    std::vector<int> shape;          // 1D: {1024};  2D: {256,256}; 3D: {8, 256, 256}
    int size;                   // Buffer size, in bytes

    Buffer(int address, int size, std::vector<int> shape) {
        this->address = address;
        this->size = size;
        for (int d=0; d<shape.size(); d++) {
            this->shape.push_back(shape[d]);
        }
    }
};
*/

class DataTransformations : tpl_nocopy<DataTransformations> {
    public:
    std::vector<TransformationNode *> transformations;

    // Helpers
    void print();
    void print(int spaces);

    // Passes - to be moved out
    void resolve_transformations();     // Collapse TXs
    bool compare_to_golden(TransformationNode * golden);
    ~DataTransformations() {
        for(auto t : transformations) {
            delete t;
        }
        transformations.clear();
    }

};


// ========================================================
//                      Helper Functions
// ========================================================

void inherit_group_attributes_from_producer(TensorPairGroup * producer, TensorPairGroup * consumer);
