#include "rose.h"
#include <string>
#include "staticSingleAssignment.h"
#include "hlkc_cache.hpp"
#include "hlk_ops.h"

// #include <functional>
#include <set>
#include <sstream>

//#include "pre.h"  // not used atm
//#include "inlinerSupport.h" // not used atm

using namespace std;
using namespace Rose;
using namespace SageBuilder;
using namespace SageInterface;

namespace fs = std::filesystem;

typedef Rose_STL_Container<SgNode*> SgNodeContainer;
typedef SgNodeContainer::iterator SgNodeContainerIter;

#define SOURCE_POSITION Sg_File_Info::generateDefaultFileInfoForTransformationNode()

HlkSfpuOp* HlkSfpuOp::make_hlk_sfpu_op(HlkSfpuOp::OpApi op_api) {
    switch(op_api) {
        case HlkSfpuOp::OpApi::hlk_sfpu_exponential: return new HlkSfpuOp_hlk_sfpu_exponential(op_api);
        case HlkSfpuOp::OpApi::hlk_sfpu_sqrt: return new HlkSfpuOp_hlk_sfpu_sqrt(op_api);
        case HlkSfpuOp::OpApi::hlk_sfpu_gelu: return new HlkSfpuOp_hlk_sfpu_gelu(op_api);
        case HlkSfpuOp::OpApi::hlk_sfpu_gelu_derivative: return new HlkSfpuOp_hlk_sfpu_gelu_derivative(op_api);
        case HlkSfpuOp::OpApi::hlk_sfpu_reciprocal: return new HlkSfpuOp_hlk_sfpu_reciprocal(op_api);
        case HlkSfpuOp::OpApi::hlk_sfpu_log: return new HlkSfpuOp_hlk_sfpu_log(op_api);
        case HlkSfpuOp::OpApi::hlk_sfpu_tanh: return new HlkSfpuOp_hlk_sfpu_tanh(op_api);
        case HlkSfpuOp::OpApi::hlk_sfpu_dropout: return new HlkSfpuOp_hlk_sfpu_dropout(op_api);
        case HlkSfpuOp::OpApi::hlk_sfpu_sigmoid: return new HlkSfpuOp_hlk_sfpu_sigmoid(op_api);
        // Do not use these ops yet, not included
        /*
        case HlkSfpuOp::OpApi::hlk_sfpu_max: return new HlkSfpuOp_hlk_sfpu_max(op_api);
        case HlkSfpuOp::OpApi::hlk_sfpu_square: return new HlkSfpuOp_hlk_sfpu_square(op_api);
        case HlkSfpuOp::OpApi::hlk_sfpu_power: return new HlkSfpuOp_hlk_sfpu_power(op_api);
        case HlkSfpuOp::OpApi::hlk_sfpu_sine: return new HlkSfpuOp_hlk_sfpu_sine(op_api);
        case HlkSfpuOp::OpApi::hlk_sfpu_cosine: return new HlkSfpuOp_hlk_sfpu_cosine(op_api);
        */
        default:
            assert(false && "Error: unsupported HlkSfpuOp");
    }
}

DeviceEnum device_enum_from_string(string device) {
    if (device.compare("grayskull") == 0) {
        return DeviceEnum::GRAYSKULL;
    } else if (device.compare("wormhole") == 0) {
        return DeviceEnum::WORMHOLE;
    } else if (device.compare("wormhole_b0") == 0) {
        return DeviceEnum::WORMHOLE_B0;
    } else {
        std::cout << "Unsupported target! " << device << std::endl;
        throw;
    }
}

SgNodeContainer find_function_calls(SgNode* node) {
    SgNodeContainer node_list = NodeQuery::querySubTree (node, V_SgFunctionCallExp);

    return node_list;
}

SgNodeContainer find_for_statements(SgNode* node) {
    SgNodeContainer node_list = NodeQuery::querySubTree (node, V_SgForStatement);

    return node_list;
}

SgFunctionSymbol* get_function_symbol(SgFunctionCallExp* func_call_exp) {
    SgExpression * func_expression = func_call_exp->get_function();
    ROSE_ASSERT(func_expression != NULL);

    SgFunctionRefExp* func_call_ref_exp = isSgFunctionRefExp(func_expression);
    ROSE_ASSERT(func_call_ref_exp != NULL);

    SgFunctionSymbol *symbol = func_call_ref_exp->get_symbol();
    ROSE_ASSERT(symbol != NULL);

    // cout << symbol->get_name().str() << endl;

    return symbol;
}

template <typename HLK_OP_T>
constexpr const char* hlk_op_type_to_string() {
    if constexpr (std::is_same<HLK_OP_T, HlkMatrixOp>::value) {
        return "HlkMatrixOp";
    } else if (std::is_same<HLK_OP_T, HlkSfpuOp>::value) {
        return "HlkSfpuOp";
    }
}

template <typename HLK_OP_T>
vector<typename HLK_OP_T::OpApi> find_hlk_ops(SgProject *project) {

    SgNodeContainer func_call_list = find_function_calls(project);

    vector<typename HLK_OP_T::OpApi> hlk_op_apis;

    for (SgNodeContainerIter iter = func_call_list.begin(); iter != func_call_list.end(); iter++) {
        SgNode* node = *iter;
        SgFunctionCallExp* func_call = isSgFunctionCallExp(node);
        ROSE_ASSERT(func_call != NULL);

        string symbol_str = get_function_symbol(func_call)->get_name().str();

        for (const auto op_api: HLK_OP_T::all_op_apis) {
            if (symbol_str == HLK_OP_T::get_op_str(op_api)) {
                if (find(hlk_op_apis.begin(), hlk_op_apis.end(), op_api) == hlk_op_apis.end()) {
                    hlk_op_apis.push_back(op_api);
                }
            }
        }
    }

    return hlk_op_apis;
}

HlkMatrixOp* get_hlk_matrix_op(SgProject *project) {
    constexpr const char* hlk_op_type_str = hlk_op_type_to_string<HlkMatrixOp>();

    vector<HlkMatrixOp::OpApi> hlk_matrix_op_apis = find_hlk_ops<HlkMatrixOp>(project);

    if (hlk_matrix_op_apis.size() == 0) {
        cout << "Error: HLK must contain an " << hlk_op_type_str << " op." << endl;
        exit(1);
    }

    if (hlk_matrix_op_apis.size() > 1) {
        cout << "Error: found multiple " << hlk_op_type_str << " ops in an HLK kernel -- not supported atm" << endl;
        exit(1);
    }

    HlkMatrixOp* hlk_matrix_op = HlkMatrixOp::make_hlk_matrix_op(hlk_matrix_op_apis[0]);
    return hlk_matrix_op;
}

vector<HlkMatrixOp*> get_hlk_matrix_ops(SgProject* project) {
    constexpr const char* hlk_op_type_str = hlk_op_type_to_string<HlkMatrixOp>();

    vector<HlkMatrixOp::OpApi> hlk_matrix_op_apis = find_hlk_ops<HlkMatrixOp>(project);

    vector<HlkMatrixOp*> ops = {};
    for (auto& api : hlk_matrix_op_apis) {
        ops.push_back(HlkMatrixOp::make_hlk_matrix_op(api));
    }

    return ops;
}

vector <HlkSfpuOp*> get_hlk_sfpu_op(SgProject *project) {
    constexpr const char* hlk_op_type_str = hlk_op_type_to_string<HlkSfpuOp>();

    vector<HlkSfpuOp::OpApi> hlk_sfpu_op_apis = find_hlk_ops<HlkSfpuOp>(project);

    //if (hlk_sfpu_op_apis.size() > 1) {
    //    cout << "Error: found multiple " << hlk_op_type_str << " ops in an HLK kernel -- not supported atm" << endl;
    //    exit(1);
    //}

    vector<HlkSfpuOp*> hlk_sfpu_ops;

    // HLK may contain an SFPU op, or it may not
    for (auto hlk_sfpu_op_api: hlk_sfpu_op_apis) {
        hlk_sfpu_ops.push_back(HlkSfpuOp::make_hlk_sfpu_op(hlk_sfpu_op_api));
    }

    return hlk_sfpu_ops;
}

vector<SgNode*> find_function_calls_by_name(SgNode* node, string func_call_name) {

    SgNodeContainer func_calls_all = find_function_calls(node);
    vector<SgNode*> func_calls;

    for (SgNodeContainerIter iter = func_calls_all.begin(); iter != func_calls_all.end(); iter++) {
        SgNode* node = *iter;
        SgFunctionCallExp* func_call = isSgFunctionCallExp(node);
        ROSE_ASSERT(func_call != NULL);

        string symbol_str = get_function_symbol(func_call)->get_name().str();
        if (symbol_str == func_call_name) {
            func_calls.push_back(func_call);
        }
    }

    return func_calls;
}

void check_hlk_main_exists(SgProject* project) {
    SgFunctionDeclaration* mainDefDecl = SageInterface::findFunctionDeclaration(project, "hlk_main", NULL, true);
    ROSE_ASSERT (mainDefDecl != NULL);

    SgFunctionDefinition* mainDef = mainDefDecl->get_definition();
    ROSE_ASSERT (mainDef != NULL);
}

SgFunctionDeclaration* find_function_declaration(SgProject* project, string func_name) {
    SgFunctionDeclaration* func_decl = SageInterface::findFunctionDeclaration(project, func_name, NULL, true);
    ROSE_ASSERT (func_decl != NULL);

    return func_decl;
}

void print_SgNodeContainer(SgNodeContainer node_list) {
    SgNodeContainer::iterator i = node_list.begin();
    while (i != node_list.end()) {
        printf ("Query node = %p = %s = %s \n",*i,(*i)->sage_class_name(),(*i)->unparseToString().c_str());
        i++;
    }
    printf("\n");
}

void remove_statement(SgNode *node) {
    SgStatement* node_statement = isSgStatement(node);
    ROSE_ASSERT(node_statement != NULL);
    SgScopeStatement* enclosing_scope = node_statement->get_scope();
    enclosing_scope->remove_statement(node_statement);
}

SgExpressionPtrList get_exp_ptr_list(SgNode *target_node) {
    SgExprListExp* args = isSgFunctionCallExp(isSgExprStatement(target_node)->get_expression())->get_args();

    ROSE_ASSERT(args != NULL);
    SgExpressionPtrList exp_ptr_list = args->get_expressions();

    return exp_ptr_list;
}

SgStatement* get_statement(SgNode *node) {
    SgStatement* statement = isSgStatement(node);
    ROSE_ASSERT(statement != NULL);
    return statement;
}

SgExprStatement* build_void_func_call_statement(string func_name, SgExpressionPtrList exp_ptr_list, SgScopeStatement *enclosing_scope) {
    SgName new_sg_name(func_name);
    SgExprListExp* exp_list = buildExprListExp(exp_ptr_list);
    SgExprStatement* call_statement = buildFunctionCallStmt(new_sg_name, buildVoidType(), exp_list, enclosing_scope);

    return call_statement;
}

SgExprStatement* build_void_func_call_statement(string func_name, SgExprListExp* exp_list, SgScopeStatement *enclosing_scope) {
    SgName new_sg_name(func_name);
    SgExprStatement* call_statement = buildFunctionCallStmt(new_sg_name, buildVoidType(), exp_list, enclosing_scope);

    return call_statement;
}

void replace_call_statement_drop_first_arg(SgNode *target_node, string new_func_name) {
    SgStatement* target_statement = get_statement(target_node);
    SgScopeStatement* enclosing_scope = target_statement->get_scope();
    SgExpressionPtrList exp_ptr_list = get_exp_ptr_list(target_node);

    SgExpressionPtrList exp_ptr_list_new;
    SgExpressionPtrList::const_iterator iter;

    // skip the first expr ("core_ptr" arg) -- not used in LLK
    iter = exp_ptr_list.begin();
    iter++;
    for (; iter != exp_ptr_list.end(); iter++) {
        exp_ptr_list_new.push_back(*iter);
    }

    SgExprStatement* new_call_statement = build_void_func_call_statement(new_func_name, exp_ptr_list_new, enclosing_scope);

    replaceStatement(target_statement, new_call_statement);
}

vector<int> get_func_call_int_arguments_value_list(SgNode *target_node, const vector<int>& arg_positions) {
    SgExpressionPtrList exp_ptr_list = get_exp_ptr_list(target_node);

    vector<int> int_arguments_value_list;
    // extract int values of arguments at positions provided by arg_positions vector
    for (const int arg_position : arg_positions) {
        SgIntVal* int_val = isSgIntVal(exp_ptr_list[arg_position]);
        if (int_val == nullptr) {
            cout << "Warning: HLK function operands did not have a compile-time constant integer for arg_position: " << arg_position << endl;
            continue;
        }
        cout << "arg value = " << int_val->get_value() << endl;
        int_arguments_value_list.push_back(int_val->get_value());
    }

    return int_arguments_value_list;
}

vector<int> get_func_call_int_arguments_value_list(SgNode *target_node, int start_position, int num_args) {
    vector<int> arg_positions;
    for (int i = start_position; i < start_position + num_args; i++) {
        arg_positions.push_back(i);
    }
    return get_func_call_int_arguments_value_list(target_node, arg_positions);
}

vector<SgExpression*> get_func_call_sg_expression_arg_list(SgNode *target_node, const vector<int>& arg_positions) {

    SgExpressionPtrList exp_ptr_list = get_exp_ptr_list(target_node);

    // extract expressions for arguments at positions provided by arg_positions vector
    vector<SgExpression*> extracted_exp_ptr_list;
    SgTreeCopy deep; // helper obj for deep copy
    for (const int arg_position : arg_positions) {
        // make a deep copy of SgExpression* (not sure if really required, possible speed gain if determined not required)
        // TODO: constrain the expression types allowed to 1) var reference (coeff), 2) ptr dereference (*coeff_ptr), 3) arrow operator (args->coeff)
        SgExpression* sg_expr_deep_copy = isSgExpression(exp_ptr_list[arg_position]->copy(deep));

        extracted_exp_ptr_list.push_back(sg_expr_deep_copy);

    }


    return extracted_exp_ptr_list;
}

void replace_hlk_call_statement(SgNode *target_node, string new_func_name, const vector<int>& arg_positions_to_keep) {
    SgStatement* target_statement = get_statement(target_node);
    SgScopeStatement* enclosing_scope = target_statement->get_scope();
    SgExpressionPtrList exp_ptr_list = get_exp_ptr_list(target_node);

    SgExpressionPtrList exp_ptr_list_new;

    // arg positions to keep specifies which args we keep as we replace the HLK call with LLK call
    // e.g., in unpack we'll skip the first expr ("core_ptr" arg) -- not used in LLK and skip the last expr (DST index arg) -- not relevant for unpack
    for (const int arg_position : arg_positions_to_keep) {
        exp_ptr_list_new.push_back(exp_ptr_list[arg_position]);
    }

    SgExprStatement* new_call_statement = build_void_func_call_statement(new_func_name, exp_ptr_list_new, enclosing_scope);

    replaceStatement(target_statement, new_call_statement);
}

#if 0
void replace_and_add_16_to_output_operand_and_drop_first_arg_in_call_statement(SgNode *target_node, string new_func_name, const int output_operand_arg_position) {
    SgStatement* target_statement = get_statement(target_node);
    SgScopeStatement* enclosing_scope = target_statement->get_scope();
    SgExpressionPtrList exp_ptr_list = get_exp_ptr_list(target_node);

    SgExpressionPtrList exp_ptr_list_new;
    SgExpressionPtrList::const_iterator iter;

    for (int i = 1; i < exp_ptr_list.size(); i++) {
        /*if(i == output_operand_arg_position) {
            SgExpression * new_exp = buildAddOp(exp_ptr_list[i], buildIntVal(16));
            exp_ptr_list_new.push_back(new_exp);
        }
        else {*/
            exp_ptr_list_new.push_back(exp_ptr_list[i]);
        //}
    }

    SgExprStatement* new_call_statement = build_void_func_call_statement(new_func_name, exp_ptr_list_new, enclosing_scope);

    replaceStatement(target_statement, new_call_statement);
}
#endif

// default insertion is to position=0 (start of func)
void insert_void_argless_func_call_to_statement_list(SgProject *project, string target_func_name, string func_call_name, int position=0) {
    SgFunctionDeclaration* main_decl = find_function_declaration(project, target_func_name);
    SgFunctionDefinition* main_def = main_decl->get_definition();
    SgBasicBlock* body = main_def->get_body();

    SgExprStatement* new_call_statement = build_void_func_call_statement(func_call_name, buildExprListExp(), main_def);

    insertStatementBefore(*(body->get_statements().begin() + position), new_call_statement);
}

void insert_void_argless_func_calls_after_func_call(SgProject *project, string parent_func_name, string target_func_name, vector<string> new_func_call_names) {
    SgFunctionDeclaration* main_decl = find_function_declaration(project, parent_func_name);
    SgFunctionDefinition* main_def = main_decl->get_definition();
    SgBasicBlock* body = main_def->get_body();

    SgNodeContainer func_call_list = find_function_calls(body);
    bool found_target = false;
    for (SgNodeContainerIter iter = func_call_list.begin(); iter != func_call_list.end(); iter++) {
        SgNode* node = *iter;
        SgFunctionCallExp* func_call = isSgFunctionCallExp(node);
        ROSE_ASSERT(func_call != NULL);

        string symbol_str = get_function_symbol(func_call)->get_name().str();
        if (symbol_str == target_func_name) {
            cout << "found target_func_name: " << target_func_name << endl;
            vector<SgStatement*>
                new_call_statements;  // need vector<SgStatement*> for the insertStatementListAfter call
            for (size_t i = 0; i < new_func_call_names.size(); i++) {
                SgExprStatement* new_call_statement =
                    build_void_func_call_statement(new_func_call_names[i], buildExprListExp(), main_def);
                new_call_statements.push_back(new_call_statement);
            }
            SgStatement* target_func_statement = isSgStatement(func_call->get_parent());

            insertStatementListAfter(target_func_statement, new_call_statements);

            found_target = true;
        }
    }
    assert(found_target && "Error: Didn't find target_func_name");
}

void insert_void_argless_func_call_to_statement_list_back(SgProject *project, string target_func_name, string func_call_name, int position=0) {
    SgFunctionDeclaration* main_decl = find_function_declaration(project, target_func_name);
    SgFunctionDefinition* main_def = main_decl->get_definition();
    SgBasicBlock* body = main_def->get_body();

    SgExprStatement* new_call_statement = build_void_func_call_statement(func_call_name, buildExprListExp(), main_def);
    insertStatementAfter(*(body->get_statements().end()-1-position), new_call_statement);
}

void insert_void_single_arg_func_call_to_statement_list(SgProject *project, string target_func_name, string func_call_name, string var_name, int position=0) {
    SgFunctionDeclaration* main_decl = find_function_declaration(project, target_func_name);
    SgFunctionDefinition* main_def = main_decl->get_definition();
    SgBasicBlock* body = main_def->get_body();

    SgExprListExp* expr_list = new SgExprListExp(SOURCE_POSITION);
    expr_list->append_expression(new SgVarRefExp(SOURCE_POSITION, main_def->lookup_variable_symbol(SgName(var_name))));

    SgExprStatement* new_call_statement = build_void_func_call_statement(func_call_name, expr_list, main_def);
    insertStatementBefore(*(body->get_statements().begin() + position), new_call_statement);
}


// void insert_void_func_call_to_statement_list_at_top_of_batch_for_loops(
//     SgProject *project,
//     string target_func_name,
//     string func_call_name,
//     SgForStatement* for_statement,
//     vector<int> operand_id_vector,
//     vector<SgExpression*> sg_expr_list,
//     int position=0
// ) {
//     SgFunctionDeclaration* main_decl = find_function_declaration(project, target_func_name);
//     SgFunctionDefinition* main_def = main_decl->get_definition();
//     SgBasicBlock* body = main_def->get_body();

//     SgExprListExp* expr_list = new SgExprListExp(SOURCE_POSITION);

//     // Create operands and assign them their operand ids for hw_configure
//     for (int operand_id: operand_id_vector) {
//         expr_list->append_expression(
//             buildIntVal(operand_id)
//         );
//     }
//     for (SgExpression* sg_expr : sg_expr_list) {
//         expr_list->append_expression(sg_expr);
//     }
//     SgExprStatement* new_call_statement = build_void_func_call_statement(func_call_name, expr_list, for_statement);
//     insertStatementBefore(for_statement, new_call_statement);
// }

void insert_void_func_call_to_statement_list(SgProject *project, string target_func_name, string func_call_name, string var_name, vector<SgExpression*> sg_expr_list, int position=0) {
    SgFunctionDeclaration* main_decl = find_function_declaration(project, target_func_name);
    SgFunctionDefinition* main_def = main_decl->get_definition();
    SgBasicBlock* body = main_def->get_body();

    SgExprListExp* expr_list = new SgExprListExp(SOURCE_POSITION);
    expr_list->append_expression(new SgVarRefExp(SOURCE_POSITION, main_def->lookup_variable_symbol(SgName(var_name))));

    for (SgExpression* sg_expr : sg_expr_list) {
        expr_list->append_expression(sg_expr);
    }

    SgExprStatement* new_call_statement = build_void_func_call_statement(func_call_name, expr_list, main_def);
    insertStatementBefore(*(body->get_statements().begin() + position), new_call_statement);
}


// TODO: ensure that this call statements are safe to remove in threads that don't emit them
// to do so, we must ensure that their argument expressions don't have side-effects -- for example they can't contain statements such as i++
// if all of them are pure statement-free expressions, we are good
void remove_call_statements(SgProject *project, const vector<string> &func_calls_to_remove) {
    SgNodeContainer func_call_list = find_function_calls(project);

    for (SgNodeContainerIter iter = func_call_list.begin(); iter != func_call_list.end(); iter++) {
        SgNode* node = *iter;
        SgFunctionCallExp* func_call = isSgFunctionCallExp(node);
        ROSE_ASSERT(func_call != NULL);

        string symbol_str = get_function_symbol(func_call)->get_name().str();
        if (find(func_calls_to_remove.begin(), func_calls_to_remove.end(), symbol_str) != func_calls_to_remove.end()) {
            cout << "Removing function call: " << symbol_str << endl;
            remove_statement(func_call->get_parent()); // parent is SgExprStatement
        }
    }
}

void replace_call_with_llk_in_main_scope(SgProject *project, SgNode *target_node, string hlk_function_name, string new_function_name, vector<int> args_to_pick) {
    SgFunctionDeclaration* main_decl = find_function_declaration(project, "hlk_main");
    SgFunctionDefinition* main_def = main_decl->get_definition();
    SgBasicBlock* body = main_def->get_body();

    SgExpressionPtrList exp_ptr_list = get_exp_ptr_list(target_node);
    SgExpressionPtrList exp_ptr_list_new;
    SgExpressionPtrList::const_iterator iter;
    for (auto arg_idx: args_to_pick) {
        if (arg_idx >= exp_ptr_list.size()) {
            continue;
        }
        iter = exp_ptr_list.begin() + arg_idx;
        for (; iter != exp_ptr_list.end(); iter++) {
            exp_ptr_list_new.push_back(*iter);
        }
    }
    SgExprStatement* new_call_statement = build_void_func_call_statement(new_function_name, exp_ptr_list_new, main_def);
    insertStatementAfter(*(body->get_statements().begin()), new_call_statement);
    remove_call_statements(project, {hlk_function_name});
}

void replace_call_statements(SgProject *project, const vector<string>& hlk_func_calls_to_replace, const vector<string>& replacement_llk_func_calls) {
    SgNodeContainer func_call_list = find_function_calls(project);

    cout << "FOUND CALLS: " << func_call_list.size() << endl;
    for (auto& c : func_call_list) {
        SgFunctionCallExp* func_call = isSgFunctionCallExp(c);
        ROSE_ASSERT(func_call != NULL);
        string symbol_str = get_function_symbol(func_call)->get_name().str();
        cout << symbol_str << endl;
    }

    for (SgNodeContainerIter iter = func_call_list.begin(); iter != func_call_list.end(); iter++) {
        SgNode* node = *iter;
        SgFunctionCallExp* func_call = isSgFunctionCallExp(node);
        ROSE_ASSERT(func_call != NULL);

        string symbol_str = get_function_symbol(func_call)->get_name().str();

        // replace wait/pop
        for (size_t i = 0; i < hlk_func_calls_to_replace.size(); i++) {
            if (hlk_func_calls_to_replace[i] == symbol_str) {
                if (hlk_func_calls_to_replace[i].find("_init_once") != std::string::npos) {
                    cout << "Inserting call: " << replacement_llk_func_calls[i] << " outside the outer-loop and removing " << symbol_str << endl;
                    replace_call_with_llk_in_main_scope(project, func_call->get_parent(), symbol_str, replacement_llk_func_calls[i], {1});
                } else {
                    cout << "Replacing call: " << symbol_str << " to " << replacement_llk_func_calls[i] << endl;
                    replace_call_statement_drop_first_arg(func_call->get_parent(), replacement_llk_func_calls[i]); // parent is SgExprStatement
                }
            }
        }
    }
}

void remove_vars_of_type(SgProject* project, const vector<string>& types, VariantT node_type) {
    std::unordered_set<SgStatement*> visited;
    for (SgNode* node : NodeQuery::querySubTree(project, node_type)) {
        SgExpression* expr = isSgExpression(node);
        SgVariableDeclaration* decl_stmt = isSgVariableDeclaration(node);
        ROSE_ASSERT(expr || decl_stmt);

        SgType* type = expr ? expr->get_type()->stripType() : decl_stmt->get_definition()->get_type()->stripType();
        ROSE_ASSERT(type);

        if (std::find(types.begin(), types.end(), type->unparseToString()) == types.end())
            continue;

        SgNode* parent_statement = node;
        while (!isSgExprStatement(parent_statement) && !isSgVariableDeclaration(parent_statement)) {
            parent_statement = parent_statement->get_parent();
        }

        SgStatement* statement = isSgStatement(parent_statement);
        ROSE_ASSERT(statement != nullptr);
        if (visited.find(statement) != visited.end()) {
            continue;
        }

        SgExpression* zero_expr = buildIntVal(0);
        SgExprStatement* void_stmt = nullptr;
        if (isSgBasicBlock(statement->get_parent())) {
            void_stmt = buildExprStatement(buildCastExp(zero_expr, buildVoidType()));
        } else {
            void_stmt = buildExprStatement(zero_expr);
        }
        cout << "Replacing variable use: " << statement->unparseToString() << " to void stmt" << endl;
        replaceStatement(statement, void_stmt);
        visited.insert(statement);
    }
}

void remove_vars_of_type(SgProject* project, const vector<string>& types) {
    remove_vars_of_type(project, types, V_SgVarRefExp);
    remove_vars_of_type(project, types, V_SgVariableDeclaration);
}

string insert_disaggregated_into_hw_configure(string hw_func_name, uint32_t relu_config=0) {
    // Find beginning of template, if any exists
    const size_t found_pos = hw_func_name.find("<");
    bool found = (found_pos != string::npos);
    bool relu_en = (relu_config&0xf)>0;
    uint32_t relu_mode = (relu_config&0xf);
    uint32_t relu_threshold = relu_config>>16;

    if(not found){
        //this base function has no templates
        hw_func_name = hw_func_name + "_disaggregated";

        if(relu_en) {
            if (relu_mode == 3) {
               hw_func_name += "<ReluType::MAX_THRESHOLD_RELU, " + to_string(relu_threshold) + ">";
            } else {
               hw_func_name += "<ReluType::MIN_THRESHOLD_RELU, " + to_string(relu_threshold) + ">";
            }
        }
    }else{
        //insert right before template starts
        hw_func_name.insert(found_pos, "_disaggregated");
        //_disaggregated should be an extension of templates found in base version
        if(relu_en){
            assert(hw_func_name.back() == '>');
            hw_func_name.pop_back();
            if (relu_mode == 3) {
               hw_func_name += ", ReluType::MAX_THRESHOLD_RELU, " + to_string(relu_threshold) + ">";
            } else {
               hw_func_name += ", ReluType::MIN_THRESHOLD_RELU, " + to_string(relu_threshold) + ">";
            }
        }
    }

    return hw_func_name;
}


void replace_hlk_inits_and_insert_hw_configure(SgProject *project, HlkMatrixOp* op, string unpack_or_pack, string dst_mode_str = "", uint32_t relu_config = 0, bool untilize_output = false) {
    assert(unpack_or_pack == "unpack" or unpack_or_pack == "pack");
    SgNodeContainer func_call_list = find_function_calls(project);
    SgFunctionDeclaration* main_decl = find_function_declaration(project, "hlk_main");
    SgFunctionDefinition* main_def = main_decl->get_definition();
    SgBasicBlock* body = main_def->get_body();

    cout << "FOUND CALLS: " << func_call_list.size() << endl;
    for (auto& c : func_call_list) {
        SgFunctionCallExp* func_call = isSgFunctionCallExp(c);
        ROSE_ASSERT(func_call != NULL);
        string symbol_str = get_function_symbol(func_call)->get_name().str();
        cout << symbol_str << endl;
    }

    cout << "Beginning search and replace" << endl;

    // All necessary for hw configure
    string op_string = op->get_op_str();
    string op_string_init = op_string + "_init";
    string op_string_init_short = op_string_init + "_short";
    string op_string_init_once = op_string_init + "_once";
    string llk_replacement_init_string;
    string hw_func_name;
    vector<int> operand_positions;
    if (unpack_or_pack == "unpack") {
        llk_replacement_init_string = op->get_unpack_init_func_str();
        hw_func_name = insert_disaggregated_into_hw_configure(op->get_unpack_hw_configure_func_str());
        operand_positions = op->get_unpack_func_hlk_api_operand_positions();
    } else {
        hw_func_name = insert_disaggregated_into_hw_configure(op->get_pack_hw_configure_func_str(), relu_config);
        llk_replacement_init_string = op->get_optional_pack_init_func_str(dst_mode_str, untilize_output);
    }

    vector<SgNode*> hlk_api_func_list_item = find_function_calls_by_name(project, op_string);

    vector<SgExpression*> unpack_hw_configure_call_expr_ptr_list;
    if (unpack_or_pack == "unpack") {
        unpack_hw_configure_call_expr_ptr_list = get_func_call_sg_expression_arg_list(
            hlk_api_func_list_item.at(0)->get_parent(),
            op->get_unpack_hw_configure_func_hlk_api_expr_positions()
        );
    }

    cout << "OP STRING: " << op_string << endl;

    // First get the operand ids for this op
    vector<int> operand_ids;
    for (SgNodeContainerIter iter = func_call_list.begin(); iter != func_call_list.end(); iter++) {
        SgNode* node = *iter;
        SgFunctionCallExp* func_call = isSgFunctionCallExp(node);
        ROSE_ASSERT(func_call != NULL);

        string symbol_str = get_function_symbol(func_call)->get_name().str();

        cout << "SYMBOL STRING: " << symbol_str << endl;
        if (op_string == symbol_str) {
            cout << "FOUND MATCH: " << symbol_str << endl;

            if (unpack_or_pack == "pack") {
                operand_ids = {16};
            } else {
                operand_ids = get_func_call_int_arguments_value_list(
                    func_call->get_parent(),
                    operand_positions
                );
                if (operand_ids.size() == 0) {
                    operand_ids = {0};
                }
            }
            break;
        }
    }

    // Find init and insert a hw configure below
    for (SgNodeContainerIter iter = func_call_list.begin(); iter != func_call_list.end(); iter++) {
        SgNode* node = *iter;
        SgFunctionCallExp* func_call = isSgFunctionCallExp(node);
        ROSE_ASSERT(func_call != NULL);

        string symbol_str = get_function_symbol(func_call)->get_name().str();

        cout << "OP STRING: " << op_string_init << ", SYMBOL STRING: " << symbol_str << endl;

        bool insert_hw_config = op_string_init_once == symbol_str;
        bool insert_op_init = op_string_init == symbol_str || op_string_init_once == symbol_str || op_string_init_short == symbol_str;

        // replace wait/pop
        // If the _once postfix exists, the llk apis are inserted outside of the outer_loop
        // If the _short postfix exists, we only insert the init api and not the configure api
        if (insert_hw_config) {
            cout << "Replacing call: " << symbol_str << " to " << llk_replacement_init_string << endl;

            SgExprListExp* expr_list = new SgExprListExp(SOURCE_POSITION);

            for (int operand_position: operand_ids) {
                expr_list->append_expression(
                    buildIntVal(operand_position)
                );
            }

            for (SgExpression* sg_expr: unpack_hw_configure_call_expr_ptr_list) {
                expr_list->append_expression(sg_expr);
            }

            SgExprStatement* new_call_statement = build_void_func_call_statement(hw_func_name, expr_list, main_def);

            // If we use _once postfix, we insert the reconfig call outside the outer_loop
            if (insert_hw_config) {
                insertStatementAfter(*(body->get_statements().begin()), new_call_statement);
            // In normal reconfig, we only replace the hlk api
            } else {
                SgStatement* func_call_statement = isSgStatement(func_call->get_parent());
                insertStatementAfter(func_call_statement, new_call_statement);
            }
        }
        if (insert_op_init) {
            if (unpack_or_pack == "unpack") {
                if (op_string_init_once == symbol_str) {
                    replace_call_with_llk_in_main_scope(project, func_call->get_parent(), op_string_init_once, llk_replacement_init_string, {1});
                } else {
                    replace_call_statement_drop_first_arg(func_call->get_parent(), llk_replacement_init_string);
                }
            } else {
                if (op_string_init_short == symbol_str) {
                    remove_call_statements(project, {op_string_init_short});
                } else if (op_string_init_once == symbol_str) {
                    replace_call_with_llk_in_main_scope(project, func_call->get_parent(), op_string_init_once, llk_replacement_init_string, {});
                } else {
                    replace_hlk_call_statement(func_call->get_parent(),llk_replacement_init_string , {});
                }
            }
        }
    }
}

void insert_hw_configure_after_init(
    SgProject *project,
    string target_function_name,
    vector<string> hw_func_names,
    vector<string> llk_init_func_names,
    vector<vector<int>> multi_op_operand_ids,
    vector<vector<SgExpression*>> multi_op_sg_expr_list
) {

    SgNodeContainer func_call_list = find_function_calls(project);
    SgFunctionDeclaration* main_decl = find_function_declaration(project, target_function_name);
    SgFunctionDefinition* main_def = main_decl->get_definition();

    int idx = 0;

    for (SgNodeContainerIter iter = func_call_list.begin(); iter != func_call_list.end(); iter++) {
        SgNode* node = *iter;
        SgFunctionCallExp* func_call = isSgFunctionCallExp(node);
        ROSE_ASSERT(func_call != NULL);

        string symbol_str = get_function_symbol(func_call)->get_name().str();

        if (llk_init_func_names.at(idx) == symbol_str) {

            vector<int> operand_ids = multi_op_operand_ids.at(idx);
            vector<SgExpression*> sg_expr_list;

            // In the case of pack, there will be no additional expressions to add, so this vector is empty
            if (multi_op_sg_expr_list.size() != 0) {
                sg_expr_list = multi_op_sg_expr_list.at(idx);
            }

            SgExprListExp* expr_list = new SgExprListExp(SOURCE_POSITION);
            cout << "Adding hw configure after : " << symbol_str << endl;

            for (int operand_id: operand_ids) {
                expr_list->append_expression(
                    buildIntVal(operand_id)
                );
            }
            for (SgExpression* sg_expr : sg_expr_list) {
                expr_list->append_expression(sg_expr);
            }

            SgStatement* func_call_statement = isSgStatement(func_call->get_parent());
            SgExprStatement* new_call_statement = build_void_func_call_statement(hw_func_names.at(idx), expr_list, main_def);
            insertStatementAfter(func_call_statement, new_call_statement);
            idx++;
            if (idx == llk_init_func_names.size()) {
                return;
            }
        }
    }

}

vector<int> replace_call_statements_for_unpack_and_get_operand_ids(SgProject *project, vector<HlkMatrixOp*> hlk_ops) {
    SgNodeContainer func_call_list = find_function_calls(project);

    vector<int> operand_ids;

    for (SgNodeContainerIter iter = func_call_list.begin(); iter != func_call_list.end(); iter++) {
        SgNode* node = *iter;
        SgFunctionCallExp* func_call = isSgFunctionCallExp(node);
        ROSE_ASSERT(func_call != NULL);

        string symbol_str = get_function_symbol(func_call)->get_name().str();

        // replace HLK OP
        for (size_t i = 0; i < hlk_ops.size(); i++) {
            if (hlk_ops[i]->get_op_str() == symbol_str) {
                cout << "found an HLK OP replacement target: " << symbol_str << endl;
                // get operand ids first
                operand_ids = get_func_call_int_arguments_value_list(
                                                        func_call->get_parent(),
                                                        hlk_ops[i]->get_unpack_func_hlk_api_operand_positions());
                // then replace
                replace_hlk_call_statement(
                                    func_call->get_parent(), // parent is SgExprStatement
                                    hlk_ops[i]->get_unpack_func_str(),
                                    hlk_ops[i]->get_unpack_func_hlk_api_args_positions_to_keep());

                break; // TODO: if we wanted to support multiple HLK calls in the HLK kernel (of the same API) we'd replace all the statements, and we could collect all operand IDs.
            }
        }
    }

    return operand_ids;
}

vector<vector<int>> replace_call_statements_for_unpack_and_get_multiple_op_operand_ids(SgProject *project, vector<HlkMatrixOp*> hlk_ops) {
    SgNodeContainer func_call_list = find_function_calls(project);

    vector<vector<int>> multiple_op_operand_ids;
    vector<int> operand_ids;

    for (SgNodeContainerIter iter = func_call_list.begin(); iter != func_call_list.end(); iter++) {
        SgNode* node = *iter;
        SgFunctionCallExp* func_call = isSgFunctionCallExp(node);
        ROSE_ASSERT(func_call != NULL);

        string symbol_str = get_function_symbol(func_call)->get_name().str();

        // replace HLK OP
        for (size_t i = 0; i < hlk_ops.size(); i++) {
            cout << "Hlk op: " << hlk_ops[i]->get_op_str() << ", symbol_str: " << symbol_str << endl;
            if (hlk_ops[i]->get_op_str() == symbol_str) {
                cout << "found an HLK OP replacement target: " << symbol_str << endl;
                // get operand ids first
                operand_ids = get_func_call_int_arguments_value_list(
                                                        func_call->get_parent(),
                                                        hlk_ops[i]->get_unpack_func_hlk_api_operand_positions());

                multiple_op_operand_ids.push_back(operand_ids);
                // then replace
                replace_hlk_call_statement(
                                    func_call->get_parent(), // parent is SgExprStatement
                                    hlk_ops[i]->get_unpack_func_str(),
                                    hlk_ops[i]->get_unpack_func_hlk_api_args_positions_to_keep());

                break; // TODO: if we wanted to support multiple HLK calls in the HLK kernel (of the same API) we'd replace all the statements, and we could collect all operand IDs.
            }
        }
    }

    return multiple_op_operand_ids;
}


void replace_hlk_dropout_with_llk_int_dropout_and_scale(SgProject *project) {

    SgFunctionDeclaration* main_decl = find_function_declaration(project, "hlk_main");
    SgFunctionDefinition* main_def = main_decl->get_definition();
    SgBasicBlock* body = main_def->get_body();

    string dropout_math_func_name = "llk_math_eltwise_unary_sfpu_dropout<true, SyncFull>";
    vector<SgNode*> dropout_func_calls = find_function_calls_by_name(project, dropout_math_func_name);

    cout << "DROPOUT FUNC CALLS SIZE: " << dropout_func_calls.size() << endl;

    int dropout_call_position = 0;
    for (SgNode* func_call: dropout_func_calls) {
        SgFunctionCallExp* dropout_func_call = isSgFunctionCallExp(func_call);
        ROSE_ASSERT(dropout_func_call != NULL);

        SgNode* target_node = dropout_func_call->get_parent();
        SgExpressionPtrList dropout_function_exp_ptr_list = get_exp_ptr_list(target_node);

        // Just get the dropout probability, which is the second argument to the llk call
        SgExpressionPtrList dropout_probability_expression_list {
            *(dropout_function_exp_ptr_list.begin() + 1)
        };

        // Build variable declaration for llk probability and llk seed
        string llk_dropout_expression_name = "llk_dropout" + to_string(dropout_call_position);
        string llk_dropout_scale_expression_name = "llk_dropout_scale" + to_string(dropout_call_position);

        SgName convert_float_to_uint16_function("convert_float_to_uint16");
        SgVariableDeclaration* llk_dropout_probability_decl = buildVariableDeclaration(
            llk_dropout_expression_name,
            buildConstType(buildAutoType()),
            buildAssignInitializer(
                buildFunctionCallStmt(
                    convert_float_to_uint16_function,
                    buildAutoType(),
                    buildExprListExp(dropout_probability_expression_list),
                    main_def->get_scope()
                )->get_expression()
            ),
            main_def
        );

        // Builds 1 / (1 - p)
         SgExpressionPtrList scale_factor_expression_list {
            buildDivideOp(
                buildFloatVal(1.0),
                buildSubtractOp(
                    buildFloatVal(1.0),
                    dropout_probability_expression_list.at(0)
                )
            ),
        };

        SgName convert_float_to_1_8_7_function("convert_float_to_1_8_7");
        SgVariableDeclaration* llk_dropout_scale_decl = buildVariableDeclaration(
            llk_dropout_scale_expression_name,
            buildConstType(buildAutoType()),
            buildAssignInitializer(
                buildFunctionCallStmt(
                    convert_float_to_1_8_7_function,
                    buildAutoType(),
                    buildExprListExp(scale_factor_expression_list),
                    main_def->get_scope()
                )->get_expression()
            ),
            main_def
        );

        insertStatementAfter(*(body->get_statements().begin()), llk_dropout_probability_decl);
        insertStatementAfter(*(body->get_statements().begin()), llk_dropout_scale_decl);

        // Create new dropout func call expression list that we will use for the llk call
        SgExpressionPtrList new_dropout_llk_func_call_expressions;

        // Add tile idx back into the new expression list
        new_dropout_llk_func_call_expressions.push_back(*(dropout_function_exp_ptr_list.begin()));

        // Add the references to the llk probability and scale into the new expression list
        new_dropout_llk_func_call_expressions.push_back(
            buildVarRefExp(
                llk_dropout_expression_name,
                main_def
            )
        );

        new_dropout_llk_func_call_expressions.push_back(
            buildVarRefExp(
                llk_dropout_scale_expression_name,
                main_def
            )
        );

        // Replace the llk dropout func call with a new func call that has this new expression list
        SgStatement* target_statement = get_statement(target_node);
        SgScopeStatement* enclosing_scope = target_statement->get_scope();
        SgExprStatement* new_call_statement = build_void_func_call_statement(dropout_math_func_name, new_dropout_llk_func_call_expressions, enclosing_scope);

        replaceStatement(target_statement, new_call_statement);

        dropout_call_position++;
    }
}

void replace_call_statements_for_math(SgProject *project, vector<HlkOp*> hlk_ops) {
    SgNodeContainer func_call_list = find_function_calls(project);

    for (SgNodeContainerIter iter = func_call_list.begin(); iter != func_call_list.end(); iter++) {
        SgNode* node = *iter;
        SgFunctionCallExp* func_call = isSgFunctionCallExp(node);
        ROSE_ASSERT(func_call != NULL);

        string symbol_str = get_function_symbol(func_call)->get_name().str();

        // replace HLK OP
        for (auto* hlk_op : hlk_ops) {
            if (hlk_op->get_op_str() == symbol_str) {
                cout << "found an HLK OP replacement target: " << symbol_str << endl;

                vector<int> hlk_op_math_args_positions_to_keep = hlk_op->get_math_func_hlk_api_args_positions_to_keep();

                replace_hlk_call_statement(
                                    func_call->get_parent(),
                                    hlk_op->get_math_func_str(),
                                    hlk_op_math_args_positions_to_keep); // parent is SgExprStatement
            }
        }
    }
}

string detect_dst_mode(SgProject* project, HlkSfpuOp* hlk_sfpu_op, HlkMatrixOp* hlk_matrix_op);

vector<int> replace_call_statements_for_pack_and_get_operand_ids(
    SgProject *project,
    HlkSfpuOp* hlk_sfpu_op,
    bool untilize_output,
    HlkMatrixOp* hlk_matrix_op,
    bool pack_microblocks,
    string device_name,
    bool fp32_dest_acc_en,
    bool pack_l1_acc_en
) {
    vector<string> hlk_call_names_to_replace = {
        "hlk_wait_for_free_tiles",
        "hlk_push_tiles",
        "hlk_pack_tile_to_stream",
        "hlk_pack_relu_tile_to_stream",
        "hlk_pack_shifted_init",
        "hlk_pack_shifted_tile_to_stream",
        "hlk_pack_shifted_relu_tile_to_stream",
    };

    vector<string> llk_names = {
        "llk_wait_for_free_tiles",
        "llk_push_tiles",
        "llk_pack",
        "llk_pack",
        "llk_pack_shifted_init",
        "llk_pack_shifted",
        "llk_pack_shifted",
    };

    vector<int> llk_operand_positions = {1, 1, 2, 2, 1, 2, 2};

    ROSE_ASSERT(hlk_call_names_to_replace.size() == llk_names.size());
    ROSE_ASSERT(llk_names.size() == llk_operand_positions.size());

    SgNodeContainer func_call_list = find_function_calls(project);

    set<int> unique_operand_ids;

    vector<int> operand_ids;

    string dst_mode = detect_dst_mode(project, hlk_sfpu_op, hlk_matrix_op);

    for (SgNodeContainerIter iter = func_call_list.begin(); iter != func_call_list.end(); iter++) {
        SgNode* node = *iter;
        SgFunctionCallExp* func_call = isSgFunctionCallExp(node);
        ROSE_ASSERT(func_call != NULL);

        string symbol_str = get_function_symbol(func_call)->get_name().str();

        // is this a replacement target?
        for (size_t i = 0; i < hlk_call_names_to_replace.size(); i++) {
            if (hlk_call_names_to_replace[i] == symbol_str) {
                // get operand ids first
                vector<int> op_id = get_func_call_int_arguments_value_list(func_call->get_parent(), llk_operand_positions[i], 1);

                cout << "found a replacement target" << endl;

                // Only one operand id currently allowed
                assert(op_id.size() == 0 or op_id.size() == 1);

                if((hlk_call_names_to_replace[i] == "hlk_pack_tile_to_stream")||
                   (hlk_call_names_to_replace[i] == "hlk_pack_relu_tile_to_stream")) //FIXME: for now treat the same as general pack
                                                                                     //but we might want to add args for relu type and threshold setting
                {
                    SgExpressionPtrList exp_ptr_list = get_exp_ptr_list(func_call->get_parent());
                    string unordered = exp_ptr_list.size() >= 4 ? "true" : "false"; // 4 arguments when tile_index is specified.
                    string untilize_str = untilize_output ? "true" : "false";
                    string replacement_name = llk_names[i] + "<" + unordered + ", " + dst_mode + ", " + untilize_str;

                    if(DeviceEnum::GRAYSKULL != device_enum_from_string(device_name)){
                        const string FP32_DEST_ACC_EN_STR = fp32_dest_acc_en ? "true" : "false";
                        replacement_name += ", " + FP32_DEST_ACC_EN_STR;
                    }

                    if(DeviceEnum::WORMHOLE_B0 == device_enum_from_string(device_name)){
                        const string PACK_L1_ACC_EN_STR = pack_l1_acc_en ? "true" : "false";
                        replacement_name += ", " + PACK_L1_ACC_EN_STR;
                    }

                    replacement_name += " >";

                    replace_call_statement_drop_first_arg(func_call->get_parent(), replacement_name);
                }
                else if(hlk_call_names_to_replace[i] == "hlk_wait_for_free_tiles")
                {
                    // TODO: How to specify only one template arg of many if positional (jchen)
                    // set template arg wait_for_blocks = true, double_buffer = false (default)
                    string untilize_str = untilize_output ? "true" : "false";
		            string pack_micro_blocks_str = pack_microblocks ? "true" : "false";
                    replace_call_statement_drop_first_arg(func_call->get_parent(), llk_names[i] + "<false," + untilize_str + "," + pack_micro_blocks_str + ">");
                }
                else if(hlk_call_names_to_replace[i] == "hlk_push_tiles")
                {
                    // set template arg push_blocks = true
                    string untilize_str = untilize_output ? "true" : "false";
		            string pack_micro_blocks_str = pack_microblocks ? "true" : "false";
                    replace_call_statement_drop_first_arg(func_call->get_parent(), llk_names[i] + "<" + untilize_str + "," + pack_micro_blocks_str + ">");
                }
                else
                {
                    replace_call_statement_drop_first_arg(func_call->get_parent(), llk_names[i]); // parent is SgExprStatement
                }

                if (op_id.size() == 1) {
                    unique_operand_ids.insert(op_id[0]);
                }
            }
        }
    }

    for(const int i : unique_operand_ids) {
        operand_ids.push_back(i);
    }

    return operand_ids;
}

vector<int> get_int_arg_vals_for_func_calls(SgProject *project, const vector<string>& target_func_call_names) {
    SgNodeContainer func_call_list = find_function_calls(project);

    vector<int> int_arg_vals_all;

    for (SgNodeContainerIter iter = func_call_list.begin(); iter != func_call_list.end(); iter++) {
        SgNode* node = *iter;
        SgFunctionCallExp* func_call = isSgFunctionCallExp(node);
        ROSE_ASSERT(func_call != NULL);

        string symbol_str = get_function_symbol(func_call)->get_name().str();

        for (size_t i = 0; i < target_func_call_names.size(); i++) {
            if (target_func_call_names[i] == symbol_str) {
                cout << "found a replacement target " << target_func_call_names[i] << endl;
                vector<int> int_arg_vals = get_func_call_int_arguments_value_list(func_call->get_parent(), 1, 1);
                int_arg_vals_all.insert(int_arg_vals_all.end(), int_arg_vals.begin(), int_arg_vals.end());
                break;
            }
        }
    }

    return int_arg_vals_all;
}

vector<int> get_int_arg_vals_for_hlk_op(SgProject *project, const HlkOp* hlk_op) {
    SgNodeContainer func_call_list = find_function_calls(project);

    vector<int> int_arg_vals;

    for (SgNodeContainerIter iter = func_call_list.begin(); iter != func_call_list.end(); iter++) {
        SgNode* node = *iter;
        SgFunctionCallExp* func_call = isSgFunctionCallExp(node);
        ROSE_ASSERT(func_call != NULL);

        string symbol_str = get_function_symbol(func_call)->get_name().str();

        if (hlk_op->get_op_str() == symbol_str) {
            cout << "extracting int arg vals from " << hlk_op->get_op_str() << endl;
            int_arg_vals = get_func_call_int_arguments_value_list(func_call->get_parent(), hlk_op->get_math_func_hlk_api_option_positions());
            cout << "extracted:  {" << endl;
            for (auto& i : hlk_op->get_math_func_hlk_api_option_positions()) {
                cout << i << endl;
            }
            cout << "}" << endl;
            break;
        }
    }

    return int_arg_vals;
}


void append_arg_to_func_decl(SgFunctionDeclaration *func_decl, string arg_name, SgType *arg_type) {
    SgName name(arg_name);
    SgInitializedName* new_arg_init_name = new SgInitializedName(name, arg_type, NULL, NULL);
    new_arg_init_name->set_file_info(SOURCE_POSITION);

    SgFunctionDefinition *func_def = func_decl->get_definition();
    ROSE_ASSERT(func_def != NULL);
    new_arg_init_name->set_scope(func_def);

    SgVariableSymbol *arg_symbol = new SgVariableSymbol(new_arg_init_name);
    func_def->insert_symbol(new_arg_init_name->get_name(), arg_symbol);

    ROSE_ASSERT(func_def->lookup_variable_symbol(new_arg_init_name->get_name()) != NULL);
    ROSE_ASSERT(new_arg_init_name->get_symbol_from_symbol_table() != NULL);

    // arg_list.push_back(llk_args_init_name); // this doesn't work, need to go the append_arg route
    ROSE_ASSERT(func_decl->get_parameterList() != NULL);
    func_decl->get_parameterList()->append_arg(new_arg_init_name);
}

void patch_main_decl_for_llk(SgProject *project, string new_main_name) {
    SgFunctionDeclaration* main_decl = find_function_declaration(project, "hlk_main");

    // change main name -- FIXME: this doesn't seem to be complete, find_fuction fails to find "unpack_main", but the code is unparsed correctly
    SgSymbolTable* symtab = main_decl->get_scope()->get_symbol_table();
    SgSymbol* sym = symtab->find(main_decl);
    if (sym) {
        symtab->remove(sym);
    }
    SgName new_sg_name(new_main_name);
    main_decl->set_name(new_sg_name);
    SgFunctionSymbol* new_symbol = new SgFunctionSymbol(main_decl);
    new_symbol->set_parent(symtab);
    symtab->insert(new_sg_name, new_symbol);

    // remove "core_ptr" (the first arg)
    SgInitializedNamePtrList& arg_list = main_decl->get_args();
    arg_list.erase(arg_list.begin());
    // TODO: should also remove the symbol

    // create a const void ptr arg for llk_args

    // create a const int arg for outer_loop_cnt
    append_arg_to_func_decl(main_decl, "outer_loop_cnt", buildConstType(buildIntType()));
}

void insert_var_decl_with_static_cast_assign_initializer(SgProject *project, string func_name, string new_var_name_str, string ref_var_name_str, SgType *type) {
    SgFunctionDeclaration* main_decl = find_function_declaration(project, func_name);
    SgFunctionDefinition* main_def = main_decl->get_definition();
    SgBasicBlock* body = main_def->get_body();

    SgVariableDeclaration* loop_iter_decl = buildVariableDeclaration(
        new_var_name_str,
        buildConstType(buildAutoType()),
        buildAssignInitializer(
            buildCastExp(buildVarRefExp(ref_var_name_str, main_def), type, SgCastExp::e_static_cast)),
        main_def);
    insertStatementBefore(*(body->get_statements().begin()), loop_iter_decl);
}

inline void insert_perf_monitor_set_flag(SgForStatement* outer_loop, string loop_iter_name) {
    SgExprListExp* expr_list_perf_target = new SgExprListExp(SOURCE_POSITION);
    expr_list_perf_target->append_expression(buildVarRefExp(loop_iter_name,outer_loop));
    string function_name = "set_perf_dump_flag_for_input";
    SgExprStatement* check_perf_dump_for_input = build_void_func_call_statement(function_name, expr_list_perf_target, outer_loop);
    appendStatement(check_perf_dump_for_input, isSgBasicBlock(outer_loop->get_loop_body()));
}

inline void insert_pack_perf_monitor_init(SgForStatement* outer_loop) {

    SgExprStatement* record_input_init_time = build_void_func_call_statement("record_pack_input_init_timestamp", buildExprListExp(), outer_loop);
    appendStatement(record_input_init_time, isSgBasicBlock(outer_loop->get_loop_body()));
}

inline void insert_pack_perf_monitor_end(SgForStatement* outer_loop) {

    SgExprStatement* record_input_end_time = build_void_func_call_statement("record_pack_input_end_timestamp", buildExprListExp(), outer_loop);
    appendStatement(record_input_end_time, isSgBasicBlock(outer_loop->get_loop_body()));
}


inline void insert_unpack_first_instruction_perf(SgForStatement* outer_loop) {

    SgExprStatement* record_input_end_time = build_void_func_call_statement("record_unpack_first_instruction_timestamp", buildExprListExp(), outer_loop);
    appendStatement(record_input_end_time, isSgBasicBlock(outer_loop->get_loop_body()));
}

inline void insert_math_perf_monitor_start(SgForStatement* outer_loop) {

    string function_name = "perf_math_counter_start";
    SgExprStatement* record_input_end_time = build_void_func_call_statement(function_name, buildExprListExp(), outer_loop);
    appendStatement(record_input_end_time, isSgBasicBlock(outer_loop->get_loop_body()));
}

inline void insert_math_perf_monitor_end(SgForStatement* outer_loop) {

    SgExprStatement* record_input_end_time = build_void_func_call_statement("record_perf_math_counter", buildExprListExp(), outer_loop);
    appendStatement(record_input_end_time, isSgBasicBlock(outer_loop->get_loop_body()));
}

void insert_outer_loop(SgProject *project, string func_name, bool insert_perf_monitor=false, int thread_id=0) {
    SgFunctionDeclaration* main_decl = find_function_declaration(project, func_name);
    SgFunctionDefinition* main_def = main_decl->get_definition();
    SgBasicBlock* body = main_def->get_body();

    string loop_iter_name = "__outer_loop_iter";
    SgVariableDeclaration* loop_iter_decl = buildVariableDeclaration(loop_iter_name, buildIntType(), NULL, main_def);
    insertStatementBefore(*(body->get_statements().begin()), loop_iter_decl);

    SgStatement* init_statement = buildAssignStatement(buildVarRefExp(loop_iter_name, main_def), buildIntVal(0));
    SgStatement* test_statement = buildExprStatement(buildLessThanOp(buildVarRefExp(loop_iter_name, main_def), buildVarRefExp("outer_loop_cnt", main_def)));
    SgExpression* incr_expression = buildPlusAssignOp(buildVarRefExp(loop_iter_name, main_def), buildIntVal(1));
    SgForStatement* outer_loop = buildForStatement(init_statement, test_statement, incr_expression, buildBasicBlock());
    insertStatementAfter(*(body->get_statements().begin()), outer_loop);

    if (insert_perf_monitor) {
        insert_perf_monitor_set_flag(outer_loop, loop_iter_name);
        if (thread_id == 1) {
            insert_math_perf_monitor_start(outer_loop);
        } else if (thread_id == 2) {
            insert_pack_perf_monitor_init(outer_loop);
        }
    }
    SgStatementPtrList main_func_statements = body->get_statements();
    // move all the original main func statements into the outer loop
    // we skip the first two statements (the two we just added): __outer_loop_iter var declaration and the outer for loop
    // all other statements are moved to the outer loop (remove & append)
    for (size_t i = 2; i < main_func_statements.size(); i++) {
        removeStatement(main_func_statements[i]);
        appendStatement(main_func_statements[i], isSgBasicBlock(outer_loop->get_loop_body()));
    }
    if (insert_perf_monitor) {
        if (thread_id == 0) {
            insert_unpack_first_instruction_perf(outer_loop);
        } else if (thread_id == 1) {
            insert_math_perf_monitor_end(outer_loop);
        } else if (thread_id == 2) {
            insert_pack_perf_monitor_end(outer_loop);
        }
    }
}

void insert_namespace(SgProject *project, string main_func_name, string namespace_name) {
    SgFunctionDeclaration* main_decl = find_function_declaration(project, main_func_name);

    SgGlobal* global_scope = isSgGlobal(main_decl->get_scope());
    ROSE_ASSERT(global_scope != NULL);

    SgName namespace_sg_name(namespace_name);
    // _nfi seems to do a more complete setup than the regular buildNamespaceDecl, however AST consitency check still complains about scope info (unparsing works though)
    // FIXME: fix the AST complaint, if it becomes a problem for unparsing
    SgNamespaceDeclarationStatement* namespace_decl = buildNamespaceDeclaration_nfi(namespace_sg_name, false, global_scope);
    //SgNamespaceSymbol* namespace_sym = new SgNamespaceSymbol(namespace_decl->get_name(), namespace_decl);
    //global_scope->insert_symbol(namespace_decl->get_name(), namespace_sym);
    //namespace_decl->set_parent(global_scope);

    global_scope->append_statement(namespace_decl);

    //SgNamespaceDefinitionStatement* namespace_def = buildNamespaceDefinition(namespace_decl);
    // namespace_def->set_global_definition(namespace_def);

    SgNamespaceDefinitionStatement* namespace_def = namespace_decl->get_definition();
    ROSE_ASSERT(namespace_def != NULL);

    SgDeclarationStatementPtrList global_scope_statements = global_scope->get_declarations();

    // move all the original global scope decl statements into the namespace
    // we skip the last statement (the one we just appended): namespace decl
    // we also skip include decl statements
    for (size_t i = 0; i < global_scope_statements.size()-1; i++) {
        removeStatement(global_scope_statements[i]);
        namespace_def->append_statement(global_scope_statements[i]);
    }
}

void insert_header(SgProject *project, string header_file_name) {
    SgFilePtrList file_list = project->get_fileList();
    //cout << "file list size = " << file_list.size() << endl;
    // only a single file should be provided to the HLKC as input
    ROSE_ASSERT(file_list.size() == 1);

    // get the one and only source file
    SgSourceFile* source_file = isSgSourceFile(project->get_fileList()[0]);
    ROSE_ASSERT(source_file != NULL);
    insertHeader(source_file, header_file_name, false, true); // false="not system header", true="as last header"
}

void remove_header(SgProject *project, string header_file_name) {

    string header_key="\""+header_file_name+"\"";

    vector<SgLocatedNode*> candidates;

    // check global scope's declarations since headers are in the global scope

    SgFilePtrList file_list = project->get_fileList();
    //cout << "file list size = " << file_list.size() << endl;
    // only a single file should be provided to the HLKC as input
    ROSE_ASSERT(file_list.size() == 1);

    SgSourceFile* source_file = isSgSourceFile(project->get_fileList()[0]);
    ROSE_ASSERT(source_file != NULL);

    SgGlobal* global= source_file -> get_globalScope();

    candidates.push_back(global);

    // check declarations within the global scope
    SgDeclarationStatementPtrList decl_stmt_list = global->get_declarations();
    for (SgDeclarationStatementPtrList::iterator iter= decl_stmt_list.begin(); iter!=decl_stmt_list.end(); iter++)
        candidates.push_back(*iter);

    bool found = false;
    for (size_t ci=0; ci<candidates.size(); ci++) {
        SgLocatedNode* locatedNode = candidates[ci];
        AttachedPreprocessingInfoType *attached_preprocessing_info = locatedNode->getAttachedPreprocessingInfo ();

        if (attached_preprocessing_info == NULL) continue;

        AttachedPreprocessingInfoType::iterator i;
        for (i = attached_preprocessing_info->begin (); i != attached_preprocessing_info->end (); i++) {
            if ((*i)->getTypeOfDirective () != PreprocessingInfo::CpreprocessorIncludeDeclaration) continue;
            string content = (*i)->getString ();
            if (content.find(header_key) != string::npos)
            {
                found = true;
                break;
            }

        } // each attached_preprocessing_info

        if (found) {
            attached_preprocessing_info->erase(i);
            break;
        }
    } // each node

    if (!found) {
        cout << "WARNING: couldn't remove " << header_file_name << ", I didn't find it." << endl;
    }
}

SgClassDeclaration* build_global_scope_struct_declaration(SgProject* project, string struct_name) {
    SgSourceFile* file = isSgSourceFile((*project)[0]);
    ROSE_ASSERT(file != NULL);
    SgClassDeclaration* struct_decl = buildStructDeclaration(struct_name, file->get_globalScope());
    return struct_decl;
}

void generate_llk_args_init_file(string llk_args_file_name, const vector<int> &operand_ids, string unpack_struct_t_str, const vector<string>& struct_member_names) {

    assert(operand_ids.size()>0 && operand_ids.size()<=struct_member_names.size() && "Error: invalid number of llk_args operands");

    ofstream file_stream;
    file_stream.open(llk_args_file_name);
    file_stream << "const " << unpack_struct_t_str << " llk_args = {" << endl;
    for (size_t i = 0; i < operand_ids.size(); i++) {
        file_stream << "\t." << struct_member_names[i] << " = (std::uint32_t) " << operand_ids[i] << ", " << endl;
    }
    file_stream << "};" << endl;
    file_stream.close();
}

void generate_llk_args_init_file_pack(string llk_args_file_name, const vector<int> &operand_ids, string unpack_struct_t_str, const vector<string>& struct_member_names) {

    assert(operand_ids.size()>0 && operand_ids.size()<=struct_member_names.size() && "Error: invalid number of llk_args operands");

    ofstream file_stream;
    file_stream.open(llk_args_file_name);
    file_stream << "const " << unpack_struct_t_str << " llk_args = {" << endl;
    for (size_t i = 0; i < operand_ids.size(); i++) {
        file_stream << "\t." << struct_member_names[i] << " = (std::uint32_t) " << operand_ids[i] << ", " << endl;
    }
    // currently generating the default Relu config --> Relu disabled.
    file_stream << "\t.relu_config = {.f = {.ApplyRelu=0, .Threshold=0}}," << endl;

    file_stream << "};" << endl;
    file_stream.close();
}


string detect_dst_mode(SgProject* project, HlkSfpuOp* hlk_sfpu_op, HlkMatrixOp* hlk_matrix_op) {

    vector<string> func_list = {"hlk_acquire_dst", "hlk_release_dst"};
    vector<int> dst_mode_int_vals = get_int_arg_vals_for_func_calls(project, func_list);

    // assert(dst_mode_int_vals.size() == 2 && "Error: there must be exactly one hlk_acquire_dst and one hlk_release_dst
    // call.");
    for (auto& mode : dst_mode_int_vals) {
        assert(
            dst_mode_int_vals[0] == mode && "Error: dst_mode must match for all hlk_acquire_dst and hlk_release_dst.");
    }
    // assert(dst_mode_int_vals[0] == dst_mode_int_vals[1] && "Error: dst_mode must match for hlk_acquire_dst and
    // hlk_release_dst.");

    const std::map<int, string> dst_mode_int_to_string = {
        {0, "SyncFull"},
        {1, "SyncHalf"},
        {2, "SyncTile16"},
    };
    string dst_mode_str = dst_mode_int_to_string.at(dst_mode_int_vals[0]);

    auto hlk_acc_to_dest = hlk_matrix_op->get_op_api() == HlkMatrixOp::OpApi::hlk_add_tile_to_dst;

    if ((dst_mode_int_vals[0] == 2) && (hlk_sfpu_op || hlk_acc_to_dest)) {
       dst_mode_str = "SyncTile2"; // Workaround for bug preventing simultaneous reads to the same bank
                                   // from sfpu and packer. SyncTile2 mode limits number of tiles in dest to 2
                                   // where each tile is stored in the separate bank. In SyncTile2 mode math/pack
                                   // will ping-pong between dest index 0 and 8 compared to SyncTile16 aka SyncTile
                                   // where all 16 indexes are used
    }
    cout << "detected dst_mode_str = " << dst_mode_str << endl;

    return dst_mode_str;
}

string detect_vector_mode(SgProject* project, HlkSfpuOp* hlk_sfpu_op) {

    string vector_mode = "RC";
    // Figure out how to determine this properly... sfpu_op or ??
    return vector_mode;
}

namespace {

struct hlk_args_recursion_context {
  stringstream file_output;
  int name_nest = 0;
  int array_nest = 0;
  vector<string> name_nest_strs = {"args"};
  set<SgClassDefinition*> struct_defs;
  vector<SgClassDefinition*> all_classes;
};

string spaces(int num_spaces) {
  string s;
  for (int i = 0; i < num_spaces; i++) {
    s += " ";
  }
  return s;
};

string build_name_nest(const vector<string>& name_nest_strs, int level) {
  string s;
  int i = 0;
  for (; i < level; i++) {
    s += name_nest_strs.at(i) + ".";
  }
  s += name_nest_strs.at(level);

  return s;
};

SgClassDefinition* get_class(const vector<SgClassDefinition*>& all_classes, string name) {
  vector<SgClassDefinition*>::const_iterator iter;
  for (iter = all_classes.begin(); iter != all_classes.end(); iter++) {
    SgClassDefinition* class_def = *iter;
    if (class_def->get_declaration()->get_name() == name) {
      return class_def;
    }
  }
  return (SgClassDefinition*)nullptr;
};

void print_all_members(hlk_args_recursion_context& ctx, string type_class_name);
void print_type(hlk_args_recursion_context& ctx, SgType* sg_type, bool parent_array = false);
void print_array_type(hlk_args_recursion_context& ctx, SgArrayType* sg_array);

// Handle Generic Type  (struct or primitive)
void print_type(hlk_args_recursion_context& ctx, SgType* sg_type, bool parent_array) {
    SgType* sg_base_type = sg_type->findBaseType();
    string base_type_str = sg_base_type->class_name();
    // cout << base_type_str << endl;
    assert(base_type_str != "SgArrayType");

    if (base_type_str == "SgClassType") {
        SgClassType* sg_class = static_cast<SgClassType*>(sg_type);
        // cout << "struct type name is: " << sg_class->get_name().getString() << endl;
        print_all_members(ctx, sg_class->get_name().getString());
  } else {
    // cout << "type name is: " << sg_type->get_name().getString() << endl;
    // cout << "type name is: " << endl;
    // str_output << "." << name_nest_strs[name_nest] << " = " << build_name_nest(name_nest) << "," << endl;
    string format_output_value;
    if (sg_base_type->isUnsignedType() && sg_base_type->isIntegerType()) {
        format_output_value = "0x\" << std::hex << (unsigned int)";
    } else if (sg_base_type->isIntegerType()) {
        format_output_value = "\" << ";
    } else if (sg_base_type->isFloatType()) {
        format_output_value = "\" << ";
    } else {
        assert("Only Float or Integer Types are supported in hlk_args_t" && false);
    }

    if (not parent_array) {
        ctx.file_output << spaces(2 * ctx.array_nest + 2) << "file_stream << \"" << spaces(2 * ctx.name_nest + 2) << "."
                        << ctx.name_nest_strs.at(ctx.name_nest) << " = " << format_output_value
                        << build_name_nest(ctx.name_nest_strs, ctx.name_nest) << " << \",\\n\"; // Output Struct Member"
                        << endl;
    } else {
        ctx.file_output << spaces(2 * ctx.array_nest + 2) << "file_stream << \"" << spaces(2 * ctx.name_nest + 2)
                        << format_output_value << build_name_nest(ctx.name_nest_strs, ctx.name_nest)
                        << " << \",\\n\"; // Output array element" << endl;
    }
  }
};

// Handle Array Type
// - check not dynamic
// - create loop for array
// - check basetype
void print_array_type(hlk_args_recursion_context& ctx, SgArrayType* sg_array) {
  assert(sg_array->get_is_variable_length_array() == 0);
  ctx.file_output << spaces(2 * ctx.array_nest + 2) << "file_stream << \"" << spaces(2 * ctx.name_nest + 2)
                  << "{\\n\"; // Array Start" << endl;

  string array_iter_str = "array_iter_" + to_string(ctx.array_nest);

  ctx.file_output << spaces(2 * ctx.array_nest + 2) << "for( int " << array_iter_str << " = 0; " << array_iter_str
                  << "<" << sg_array->get_number_of_elements() << "; " << array_iter_str << "++) {" << endl;
  ctx.array_nest++;
  string base_type_str = sg_array->get_base_type()->class_name();
  cout << base_type_str << endl;

  ctx.name_nest_strs[ctx.name_nest] += "[" + array_iter_str + "]";
  if (base_type_str == "SgArrayType") {
    print_array_type(ctx, static_cast<SgArrayType*>(sg_array->get_base_type()));
    ctx.file_output << spaces(2 * ctx.array_nest + 2) << "file_stream << \"" << spaces(2 * ctx.name_nest + 2)
                      << ",\\n\"; // Assignment to array member done" << endl;
  } else {
    print_type(ctx, sg_array->get_base_type(), true);
  }
  ctx.array_nest--;
  // str_output << spaces(2*ctx.array_nest + 2) << "}" << endl;
  ctx.file_output << spaces(2 * ctx.array_nest + 2) << "}" << endl;

  ctx.file_output << spaces(2 * ctx.array_nest + 2) << "file_stream << \"" << spaces(2 * ctx.name_nest + 2)
                  << "}\\n\"; // Array End" << endl;
};

// struct type
// - iterate through members
// - check each type, get basetype
//   - no ptrs
void print_all_members(hlk_args_recursion_context& ctx, string type_class_name) {
  ctx.file_output << spaces(2 * ctx.array_nest + 2) << "file_stream << \"" << spaces(2 * ctx.name_nest + 2)
                  << "{\\n\"; // Struct Start" << endl;
  ctx.name_nest++;
  SgClassDefinition* class_def = get_class(ctx.all_classes, type_class_name);
  ctx.struct_defs.insert(class_def);
  SgDeclarationStatementPtrList member_declarations = class_def->get_members();
  for (SgDeclarationStatementPtrList::const_iterator iter_member_decl = member_declarations.begin();
       iter_member_decl != member_declarations.end();
       iter_member_decl++) {
    SgDeclarationStatement* member_decl = *iter_member_decl;

    // static members are not assignable at runtime
    if (isStatic(member_decl)) {
        continue;
    }

    SgVariableDeclaration* member_var_decl = isSgVariableDeclaration(member_decl);

    assert(member_var_decl != NULL);

    Rose_STL_Container<SgInitializedName*>& var_list = member_var_decl->get_variables();
    assert(var_list.size() == 1);  // ROSE docs says that this will currently be 1...
    // Rose_STL_Container<SgInitializedName*>::iterator var = var_list.begin();

    SgInitializedName* sg_name = var_list[0];
    string member_var_str = sg_name->get_name().getString();
    string member_type_str = sg_name->get_type()->class_name();

    // cout << i++ << ": " << member_type_str << " : " << member_var_str << endl;
    if (ctx.name_nest_strs.size() <= ctx.name_nest) {
      ctx.name_nest_strs.resize(ctx.name_nest + 1);
    }
    ctx.name_nest_strs.at(ctx.name_nest) = member_var_str;
    // str_output << "level: " << name_nest << ", " << name_nest_strs.at(name_nest) << endl;
    // file_output << "file_stream << \"." << member_var_str << " = \" <<" << build_name_nest(array_nest) << " << " <<
    // member_var_str << " << array_indexing << " << \",\\n\";" << endl;
    if (member_type_str == "SgArrayType") {
      SgArrayType* sg_array = static_cast<SgArrayType*>(sg_name->get_type());
      // str_output << "." << name_nest_strs[name_nest] << " = " << build_name_nest(name_nest) << "," << endl;
      ctx.file_output << spaces(2 * ctx.array_nest + 2) << "file_stream << \"" << spaces(2 * ctx.name_nest + 2) << "."
                      << ctx.name_nest_strs[ctx.name_nest] << " =\\n\"; // Assgin to this array member" << endl;
      print_array_type(ctx, sg_array);
      ctx.file_output << spaces(2 * ctx.array_nest + 2) << "file_stream << \"" << spaces(2 * ctx.name_nest + 2)
                      << ",\\n\"; // Assignment to array memeber done" << endl;
    } else {
      print_type(ctx, sg_name->get_type());
    }
  }
  ctx.name_nest--;
  if (ctx.name_nest) {
    ctx.file_output << spaces(2 * ctx.array_nest + 2) << "file_stream << \"" << spaces(2 * ctx.name_nest + 2)
                    << "},\\n\"; // Struct End" << endl;
  } else {
    ctx.file_output << spaces(2 * ctx.array_nest + 2) << "file_stream << \"" << spaces(2 * ctx.name_nest + 2)
                    << "};\\n\"; // Struct End" << endl;
  }
  // cout << "END PRINT MEMBERS: " << type_class_name << endl;
};

}  // namespace

void generate_hlk_args_struct_init_generator(SgProject *project, string output_file_name) {

    // insert the generic LLK math header
    remove_header(project, "compute_hlk_api.h");

    SgFunctionDeclaration* main_decl = find_function_declaration(project, "hlk_main");
    SgGlobal* global_scope = isSgGlobal(main_decl->get_scope());
    ROSE_ASSERT(global_scope != NULL);

    // Create a parameter list with a parameter
    SgName args_name = "args";

    SgClassSymbol *hlk_args_t_sym = global_scope->lookup_class_symbol(SgName("hlk_args_t"));
    ROSE_ASSERT(hlk_args_t_sym != NULL);

    SgReferenceType *ref_type = buildReferenceType(buildConstType(hlk_args_t_sym->get_type()));
    SgInitializedName *args_init_name = buildInitializedName(args_name, ref_type);
    SgFunctionParameterList* parameterList = buildFunctionParameterList();
    appendArg(parameterList, args_init_name);

    // Create a defining functionDeclaration (with a function body)
    SgName func_name             = "generate_hlk_args_t_init";
    SgFunctionDeclaration* func  = buildDefiningFunctionDeclaration(func_name, buildIntType(), parameterList,global_scope);
    SgBasicBlock* func_body      = func->get_definition()->get_body();

    // Insert a statement in the function body
    SgVarRefExp *var_ref = buildVarRefExp(args_name,func_body);
    SgPlusPlusOp *pp_expression = buildPlusPlusOp(var_ref);
    SgExprStatement* new_stmt = buildExprStatement(pp_expression);

    // insert a statement into the function body
    prependStatement(new_stmt,func_body);
    appendStatement(func,global_scope);

    // find the class definition brute force, we can't get it through the symbol
    vector<SgClassDefinition*> all_classes =
        SageInterface::querySubTree<SgClassDefinition>(project, V_SgClassDefinition);

    hlk_args_recursion_context ctx;
    ctx.all_classes = all_classes;
    ::print_all_members(ctx, "hlk_args_t");

    ofstream file_stream;
    file_stream.open(output_file_name);

    // Alright, since nice savage approach didn't work, we're gonna do true savage now.
    file_stream << endl;

    file_stream << "#include <cstdint>" << endl;
    file_stream << "#include <fstream>" << endl << endl;

    // unparse the struct defs
    for (auto& def : ctx.struct_defs) {
      file_stream << def->unparseToString() << ";" << endl << endl;
    }

    // generate the struct generator
    file_stream << "extern \"C\" void generate_hlk_args_struct_init(const void* args_ptr, const char *out_file_name) {" << endl;
	file_stream << "\tstd::ofstream file_stream;\n";
	file_stream << "\tfile_stream.open(out_file_name);\n\n";
	file_stream << "\thlk_args_t args = *((hlk_args_t*)args_ptr);\n\n";

    file_stream << "\tfile_stream << \"const NAMESPACE::hlk_args_t hlk_args = \\n\";" << endl;
    file_stream << ctx.file_output.str();

    file_stream << "}" << endl;

    file_stream.close();
}

void get_live_variables_for_loop(LivenessAnalysis * liv, SgNode* loop, std::vector<SgInitializedName*>& live_ins, std::vector<SgInitializedName*>& live_outs)
{
  ROSE_ASSERT(liv != NULL);
  ROSE_ASSERT(loop!= NULL);

  // For SgForStatement, virtual CFG node which is interesting has an index number of 2,
  // as shown in its dot graph's node caption.
  // "<SgForStatement> @ 8: 2" means this node is for a for statement at source line 8, with an index 2.
  CFGNode cfg_node(loop, isSgForStatement(loop) ? 2 : 1);
  FilteredCFGNode<IsDFAFilter> filter_node = FilteredCFGNode<IsDFAFilter> (cfg_node);

  // Check edges
  vector<FilteredCFGEdge<IsDFAFilter>> out_edges = filter_node.outEdges();
  ROSE_ASSERT(out_edges.size()==2);
  vector<FilteredCFGEdge<IsDFAFilter>>::iterator iter = out_edges.begin();

  for (; iter!=out_edges.end();iter++)
  {
    FilteredCFGEdge < IsDFAFilter > edge= *iter;
    // one true edge going into the loop body
    //x. Live-in (loop) = live-in (first-stmt-in-loop)
    if (edge.condition()==eckTrue)
    {
      SgNode* firstnode= edge.target().getNode();
      live_ins = liv->getIn(firstnode);
    }
    // one false edge going out of loop
    //x. live-out(loop) = live-in (first-stmt-after-loop)
    else if (edge.condition()==eckFalse)
    {
      SgNode* firstnode= edge.target().getNode();
      live_outs = liv->getIn(firstnode);
    }
    else
    {
      cerr<<"Unexpected CFG out edge type for SgForStmt!"<<endl;
      ROSE_ASSERT(false);
    }
  } // end for (edges)

}

DefUseAnalysis* create_and_run_def_use_analysis(SgProject *project) {
    DefUseAnalysis* def_use = new DefUseAnalysis(project);
    bool debug = false;
    def_use->run(debug);
    if (debug) {
        def_use->dfaToDOT();
     }
    return def_use;
}

LivenessAnalysis* create_and_run_liveness_analysis(SgProject *project, DefUseAnalysis *def_use) {
    bool debug = false;
    LivenessAnalysis* liv = new LivenessAnalysis(debug, def_use);
    ROSE_ASSERT (liv != NULL);

    // find all function definitions
    Rose_STL_Container<SgNode*> func_node_list= NodeQuery::querySubTree(project, V_SgFunctionDefinition);
    std::vector <FilteredCFGNode < IsDFAFilter > > dfa_functions;
    bool abortme = false;

    for (Rose_STL_Container<SgNode*>::const_iterator i = func_node_list.begin(); i!=func_node_list.end(); ++i)
    {
        SgFunctionDefinition* func = isSgFunctionDefinition(*i);
        // run liveness analysis on func
        FilteredCFGNode <IsDFAFilter> rem_source = liv->run(func,abortme);

        if (abortme) {
            assert (false && "Error: Liveness analysis is ABORTING.");
        }
        if (rem_source.getNode() != NULL) {
            dfa_functions.push_back(rem_source);
        }
    }

    if (debug) {
        SgFilePtrList file_list = project->get_fileList();
        std::string firstFileName = StringUtility::stripPathFromFileName(file_list[0]->getFileName());
        std::string fileName = firstFileName+"_liveness.dot" ;
        std::ofstream fs(fileName.c_str());
        dfaToDot(fs, string("var"), dfa_functions, def_use, liv);
        fs.close();
    }

    return liv;
}

StaticSingleAssignment* create_and_run_ssa_analysis(SgProject *project) {
    StaticSingleAssignment* ssa = new StaticSingleAssignment(project);
    ssa->run(false, true); // no interprocedural, treat ptrs as structs

    return ssa;
}

vector<SgStatement*> dead_for_loop_analysis(SgProject* project, LivenessAnalysis* liv, StaticSingleAssignment* ssa) {
    Rose_STL_Container<SgNode*> loop_list;
    Rose_STL_Container<SgNode*> for_loop_list = NodeQuery::querySubTree(project, V_SgForStatement);
    Rose_STL_Container<SgNode*> while_loop_list = NodeQuery::querySubTree(project, V_SgWhileStmt);
    loop_list.insert(loop_list.end(), for_loop_list.begin(), for_loop_list.end());
    loop_list.insert(loop_list.end(), while_loop_list.begin(), while_loop_list.end());
    printf("\n loop_list.size() = %zu \n", loop_list.size());

    vector<SgStatement*> dead_loops;

    for (Rose_STL_Container<SgNode*>::iterator it = loop_list.begin(); it != loop_list.end(); it++) {
        bool loop_is_dead = true;

        printf ("\nQuery node = %p = %s = %s \n",*it,(*it)->sage_class_name(),(*it)->unparseToString().c_str());
        SgStatement* loop = isSgStatement(*it);
        assert(loop);
        assert(isSgForStatement(loop) || isSgWhileStmt(loop));

        SgNodeContainer func_list = find_function_calls(loop);
        if (func_list.size() > 0) {
            loop_is_dead = false;
            cout << "Loop is not dead, it has " << func_list.size() << " func calls." << endl;
            //continue; // for debug info debug we can keep going
        } else {
            cout << "Loop has no function calls, a candidate for bye-bye-loop." << endl;
        }

        std::vector<SgInitializedName*> live_ins, live_outs;
        get_live_variables_for_loop(liv, loop, live_ins, live_outs);

        cout << "live ins: ";
        for (size_t i = 0; i < live_ins.size(); i++) {
            SgInitializedName* init_name = live_ins[i];
            cout << init_name->get_name().str() << ", ";
        }
        cout << endl << "live outs: ";
        for (size_t i = 0; i < live_outs.size(); i++) {
            SgInitializedName* init_name = live_outs[i];
            cout << init_name->get_name().str() << ", ";
        }
        cout << endl;

        // alternate method for live_in/live_out, not used
#if 0
        std::set<SgInitializedName*> liveIns;
        std::set<SgInitializedName*> liveOuts;
        std::set<SgInitializedName*> live_diff;

        getLiveVariables(liv, loop, liveIns, liveOuts);
        cout << "live ins: ";
        for (SgInitializedName* init_name : liveIns) {
        cout << init_name->get_name().str() << ", ";
        }
        cout << endl << "live outs: ";
        for (SgInitializedName* init_name : liveOuts) {
        cout << init_name->get_name().str() << ", ";
        }
        cout << endl;

        set_difference(liveOuts.begin(), liveOuts.end(),
                        liveIns.begin(), liveIns.end(),
                        inserter(live_diff, live_diff.begin()));
        cout << "live_diff size: " << live_diff.size() << endl;
#endif

        // we could skip if live-outs are == 0, for debug we do it anway
        typedef StaticSingleAssignment::VarName VarName;
        std::set<VarName> var_names_defined_in_for_loop = ssa->getVarsDefinedInSubtree(loop);
        vector<SgInitializedName*> def_and_live_outs;

        cout << "defined in the loop: ";
        for (const VarName var_name : var_names_defined_in_for_loop) {
            cout << StaticSingleAssignment::varnameToString(var_name) << ", ";
            //for (const SgInitializedName* init_name : var_name) {
            //    cout << init_name->get_name().str() << ", ";
            //}
            if (var_name.size() > 1) {
                loop_is_dead = false;
                cout << "var_name.size() = " << var_name.size() << ", cannot handle this case, thus loop is not dead." << endl;
                //break; // keep going
            }
            SgInitializedName* init_var_name = var_name[0];
            if (find(live_outs.begin(), live_outs.end(), init_var_name) != live_outs.end()) {
                def_and_live_outs.push_back(init_var_name);
            }
        }

        if (def_and_live_outs.size() > 0) {
            loop_is_dead = false;
            cout << endl << "loop is not dead! defined in the loop, and also live-out: ";
            for (const SgInitializedName* def_and_live_out : def_and_live_outs) {
                cout << def_and_live_out->get_name().str() << ", ";
            }
        }

        if (loop_is_dead) {
            cout << endl << "loop is dead!" << endl;
            dead_loops.push_back(loop);
        }
        cout << endl;
    }

    return dead_loops;
}

// it will eliminate dead for-loops
// can be extended to support while, do-while and assignments
// for-loops are common-case/top-priority
void dead_for_loop_elimination(SgProject *project) {
    DefUseAnalysis* def_use = create_and_run_def_use_analysis(project);
    LivenessAnalysis* liv = create_and_run_liveness_analysis(project, def_use);
    StaticSingleAssignment* ssa = create_and_run_ssa_analysis(project);

    vector<SgStatement*> dead_loops = dead_for_loop_analysis(project, liv, ssa);

    // reverse direction elimination
    for (int i = dead_loops.size()-1; i >=0; i--) {
        cout << i << endl;
        removeStatement(dead_loops[i]);
    }
}
