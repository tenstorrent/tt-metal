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

using namespace std;


// ========================================================
//                      CLASSES
// ========================================================

class TensorData {
    public:
        vector<int> data;
        int rank;
        vector<int> shape;
        int volume;

        TensorData(vector<int> shape);

        void init_increasing();
        void print();
        void generate_csv(string filename);
};

class DTXTensor {
    public:
        vector<int> str;
        vector<int> end;
        int rank = -1;

        DTXTensor(vector<int> str, vector<int> end){
            this->str = str;
            this->end = end;
            this->rank = this->str.size();
        }

        DTXTensor() {
            this->rank = 0;
        };

        int volume();

        void print();
        string get_string();
};

class TensorPair {
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

        string get_string();
        string get_short_string();
        void print_string();
        ~TensorPair() {
            if(src_tensor) {
                delete src_tensor;
            }
            if(dst_tensor) {
                delete dst_tensor;
            }
        }
};

class Transfer {
    public:
    int src_address;
    int dst_address;
    int size;
    int pad;
    vector<int> src_soc_core;

    string get_string();

};

class TensorPairGroup {
    public:
    vector<int> shape;
    vector<TensorPair *> tensor_pairs;

    vector<Transfer *> transfers;

    // Attributes
    int address;                // Address of buffer this is stored in (L1 or DRAM)
    vector<int> core = {-1,-1}; // Tensix core
    int streaming_id;           // The sequence order that's loaded into a CB or Buffer
    ~TensorPairGroup() {
        for(auto tp : tensor_pairs) {
            if(tp)
                delete tp;
        }
        tensor_pairs.clear();
        for(auto tr : transfers) {
            if(tr)
                delete tr;
        }
        transfers.clear();
    }
};

class TransformationNode {
    public:

    string opcode;
    vector<TensorPairGroup *> groups;

    TransformationNode(string opcode, int number_of_groups) {
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

    ~TransformationNode() {
        for(auto g : groups) {
            if (g)
                delete g;
        }
        groups.clear();
    }
};



/*
class Buffer {
    public:

    int address;
    vector<int> shape;          // 1D: {1024};  2D: {256,256}; 3D: {8, 256, 256}
    int size;                   // Buffer size, in bytes

    Buffer(int address, int size, vector<int> shape) {
        this->address = address;
        this->size = size;
        for (int d=0; d<shape.size(); d++) {
            this->shape.push_back(shape[d]);
        }
    }
};
*/

class DataTransformations {
    public:
    vector<TransformationNode *> transformations;

    // Helpers
    void print();
    void print(int spaces);

    // Passes - to be moved out
    void resolve_transformations();     // Collapse TXs
    bool compare_to_golden(TransformationNode * golden);

};


// ========================================================
//                      Helper Functions
// ========================================================

void inherit_group_attributes_from_producer(TensorPairGroup * producer, TensorPairGroup * consumer);
