#pragma once

#include <cstdint>

namespace tt {

namespace tt_metal {

class red_black_tree {
   public:
    struct node_t {
        enum class colour_t {
            BLACK = 0,
            RED = 1
        };

        node_t *parent = nullptr;
        node_t *left = nullptr;
        node_t *right = nullptr;
        colour_t colour;
        uint32_t key;
    };

    red_black_tree() : root_(nullptr) {}

    red_black_tree(uint32_t key);

    node_t *search(uint32_t key);

    node_t *search_best(uint32_t key);

    node_t *successor(node_t *node);

    node_t *insert(uint32_t key);

    void remove(node_t *node_to_remove);

   private:
    node_t *init_root(uint32_t key);

    void left_rotate(node_t *node);

    void right_rotate(node_t *node);

    node_t *insert(node_t *root_node, uint32_t key);

    void fix_insertion(node_t *inserted_node);

    void transplant(node_t *node_x, node_t *node_y);

    void fix_removal(node_t *node);

    node_t *root_;
};

}  // namespace tt_metal

}  // namespace tt
