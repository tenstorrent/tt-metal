#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

// ========================================================
//                      CLASSES
// ========================================================

int Tensor::volume() {
    if (this->rank == 0 || this->rank == -1) return 0;
    
    int volume = 1;
    for (int d=0; d<this->rank; d++) {
        assert(this->end[d] >= this->str[d]);
        int dim_size = this->end[d] - this->str[d] + 1;
        volume = volume * dim_size;
    }
    return volume;
}

void Tensor::print() {
    cout << this->get_string() << endl;
}

string Tensor::get_string() {
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
        
        cout << s(4+spaces) << "TensorPairs:" << endl;
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


