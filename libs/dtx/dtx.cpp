#include <iostream>
#include <fstream>


#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

// ========================================================
//                      CLASSES
// ========================================================

TensorData::TensorData(vector<int> shape) {
    this->shape = shape;
    this->rank = shape.size();
    this->volume = vector_product(shape);

    for (int i=0; i<this->volume; i++){
        this->data.push_back(i);
    }
}

void TensorData::print() {
    bool DEBUG = true;
    if (DEBUG) cout << "Printing TensorData " << endl;

    vector<int> counter = zeros(this->rank);
    for (int i=0; i<this->volume; i++){

        if (DEBUG) cout << s(2) << "i = " << i << ".  counter = " << v2s(counter) << endl;

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
    bool DEBUG = true;

    string full_filename;
    full_filename.append(filename);
    full_filename.append(".csv");
    ofstream myfile(full_filename);


    if (DEBUG) cout << "Generating csv file: " << full_filename << endl;

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
    cout << this->get_string() << endl;
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
    cout << this->get_string() << endl;
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

    cout << s(spaces) << "Transformation Node: opcode = " << this->opcode << endl;

    int group_index = 0;
    for (TensorPairGroup * group : this->groups) {

        cout << s(2 + spaces) << "Group = " << group_index << ";  shape = " << v2s(group->shape) << ", core=" << v2s(group->core) << endl;

        //cout << s(4+spaces) << "TensorPairs:" << endl;
        cout << s(4+spaces) << "TensorPairs (" << group->tensor_pairs.size() << "):" << endl;
        int tp_index = 0;
        for (TensorPair * tp : group->tensor_pairs) {
            cout << s(6+spaces) << "TensorPair[" << tp_index << "]  " << tp->get_string() << endl;
            tp_index++;
        }

        cout << s(4+spaces) << "Transactions:" << endl;
        int tx_index = 0;
        for (Transfer * tx : group->transfers) {
            cout << s(6+spaces) << "Transactoin[" << tx_index << "]  " << tx->get_string() << endl;
            tx_index++;
        }
        group_index++;
    }
    cout << endl;
}

void DataTransformations::print() {
    this->print(0);
}

void DataTransformations::print(int spaces) {
    cout << "\n" << endl;
    cout << s(spaces) << "DataTransformations -- nodes = " << this->transformations.size() << endl;
    cout << s(spaces) << "----------------------------------------------------\n" << endl;
    for (int t=0; t<this->transformations.size(); t++) {
        this->transformations[t]->print(spaces+3);
    }
}





bool DataTransformations::compare_to_golden(TransformationNode * golden) {
    bool pass = true;
    return true;
}
