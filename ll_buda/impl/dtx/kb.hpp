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

//================================================
//               KERNEL BUFFER GRAPH
//================================================

enum NodeType {
    Buffer_L1 = 0,
    Buffer_DRAM = 0,
    CB_L1 = 1,
    CB_DRAM = 1,
    DataMovementKernel = 2,
    ComputeKernel = 3,
};

class Node {
    public:

    // Common
    int id;
    string name;
    NodeType type;
    vector<Node *> inputs;
    vector<Node *> outputs;
    vector<int> core;
    
    // Kernel
    string kernel_path;

    // DRAM buffer
    int size;               // size in Bytes
    int address;            // L1 and dram
    int dram_channel;       // dram channel

    Node() {};

};

class Graph {
    public: 
    string name;
    vector<Node*> nodes;

    void add_node(Node * new_node) {this->nodes.push_back(new_node);}

    void add_edge(Node * producer, Node * consumer) {
        producer->inputs.push_back(consumer);
        consumer->outputs.push_back(producer);
    }

    Node * create_buffer_dram(string name, int dram_channel) {
        Node * buf = new Node();
        this->nodes.push_back(buf);
        buf->type = NodeType::Buffer_DRAM;
        buf->name = name;
        buf->dram_channel = dram_channel;
        return buf;
    }

    Node * create_buffer_l1(string name, int size, vector<int> core) {
        Node * buf = new Node();
        this->nodes.push_back(buf);
        buf->type = NodeType::Buffer_L1;
        buf->name = name;
        buf->size = size;
        buf->core.push_back(core[0]);
        buf->core.push_back(core[1]);
        return buf;
    }

    Node * create_data_movement_kernel(string name, string path, vector<int> core) {
        Node * kernel = new Node();
        this->nodes.push_back(kernel);
        kernel->name = name;
        kernel->type = NodeType::DataMovementKernel;
        kernel->core.push_back(core[0]);
        kernel->core.push_back(core[1]);
        return kernel;
    }

    Node * create_compute_kernel() {
        return nullptr;
    }

};


/*
class Graph {

};



class TensorMap {

};

class Transaction {
    
    // Source
    pair<int, int> src_soc_location;
    int src_address;
    int src_size;
    
    // Destination
    pair<int, int> src_soc_location;
    int src_address;
    int src_size;
};

class Node {    
    NodeType node_type;
    vector<Core> cores;
    
    TensorMap tensor_map;
    vector<Transaction> transaction_list;

};


//================================================
//               KERNEL BUFFER GRAPH
//================================================

enum CoreType {
    dram_bank = 0,
    tensix = 1,
};



enum HardwareTarget {
    dram_bank=0,
    l1=1,
    risc0=2,        // data movement kernel
    risc1=3,        // data movement kernel
    compute=4,      // math engine, vector engine, unpack, pack
};

class Core {
    pair<int, int> soc_location;
    


};

*/