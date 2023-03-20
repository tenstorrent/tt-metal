#include "tt_metal/impl/memory_manager/red_black_tree.hpp"

#include <iostream>

namespace tt {

namespace tt_metal {

red_black_tree::red_black_tree(uint32_t key) {
    this->root_ = init_root(key);
}

red_black_tree::node_t *red_black_tree::init_root(uint32_t key) {
    return new node_t{
        .parent=nullptr,
        .left=nullptr,
        .right=nullptr,
        .colour=node_t::colour_t::BLACK,
        .key=key
    };
}

red_black_tree::node_t *red_black_tree::successor(node_t *node) {
    node = node->right;
	while (node->left != nullptr) {
		node = node->left;
	}
	return node;
}

void red_black_tree::left_rotate(node_t *node) {
    node_t *node_parent = node->parent;
	node_t *right_child = node->right;
	node->right = right_child->left;
	if (right_child->left != nullptr) {
        right_child->left->parent = node;
    }
	right_child->parent = node_parent;
	if (node_parent == nullptr) {
        this->root_ = right_child;
    }
	else if (node_parent->left == node) {
        node_parent->left = right_child;
    }
	else {
        node_parent->right = right_child;
    }
	right_child->left = node;
	node->parent = right_child;
}

void red_black_tree::right_rotate(node_t *node) {
    node_t *node_parent = node->parent;
	node_t *left_child = node->left;
	node->left = left_child->right;
	if (left_child->right != nullptr) {
        left_child->right->parent = node;
    }
	left_child->parent = node_parent;
	if (node_parent == nullptr) {
        this->root_ = left_child;
    }
	else if (node_parent->left == node) {
        node_parent->left = left_child;
    }
	else {
        node_parent->right = left_child;
    }
	left_child->right = node;
	node->parent = left_child;
}

red_black_tree::node_t *red_black_tree::search(uint32_t key) {
    node_t *node = this->root_;
	while (node != nullptr) {
		if (node->key == key) {
            break;
        } else if (key < node->key) {
            node = node->left;
        }
		else {
            node  = node->right;
        }
	}
	return node;
}

red_black_tree::node_t *red_black_tree::search_best(uint32_t key) {
    node_t *found_node = nullptr;
    node_t *curr_node = this->root_;
    while (curr_node != nullptr) {
        found_node = curr_node;
        if (curr_node->key == key) {
            break;
        } else if (curr_node->key < key) {
            curr_node = curr_node->right;
        } else {
            curr_node = curr_node->left;
        }
    }
    while (found_node != nullptr and key > found_node->key) {
        found_node = found_node->parent;
    }
    return found_node;
}

red_black_tree::node_t *red_black_tree::insert(node_t *root_node, uint32_t key) {
    if (root_node == nullptr) {
        return new node_t{.colour=node_t::colour_t::RED, .key=key};
    }

    if (key < root_node->key) {
        node_t *left_child = insert(root_node->left, key);
        root_node->left = left_child;
        left_child->parent = root_node;
    } else if (key > root_node->key) {
        node_t *right_child = insert(root_node->right, key);
        root_node->right = right_child;
        right_child->parent = root_node;
    }
    return root_node;
}

void red_black_tree::fix_insertion(node_t *inserted_node) {
    node_t *node = inserted_node;
    while (node != this->root_ and node->parent->colour == node_t::colour_t::RED) {
        if (node->parent == node->parent->parent->left) {
            node_t *gp_right = node->parent->parent->right;
            if (gp_right->colour == node_t::colour_t::RED) {
                node->parent->colour = node_t::colour_t::BLACK;
                gp_right->colour = node_t::colour_t::BLACK;
                node->parent->parent->colour = node_t::colour_t::RED;
                node = node->parent->parent;
            } else {
                if (node == node->parent->right) {
                    node = node->parent;
                    left_rotate(node);
                }
                node->parent->colour = node_t::colour_t::BLACK;
                node->parent->parent->colour = node_t::colour_t::RED;
                right_rotate(node->parent->parent);

            }
        } else {
            node_t *gp_left = node->parent->parent->left;
            if (gp_left->colour == node_t::colour_t::RED) {
                node->parent->colour = node_t::colour_t::BLACK;
                gp_left->colour = node_t::colour_t::BLACK;
                node->parent->parent->colour = node_t::colour_t::RED;
                node = node->parent->parent;
            } else {
                if (node == node->parent->left) {
                    node = node->parent;
                    right_rotate(node);
                }
                node->parent->colour = node_t::colour_t::BLACK;
                node->parent->parent->colour = node_t::colour_t::RED;
                left_rotate(node->parent->parent);
            }
        }
    }
    this->root_->colour = node_t::colour_t::BLACK;
}

red_black_tree::node_t *red_black_tree::insert(uint32_t key) {
    if (this->root_ == nullptr) {
        this->root_ = init_root(key);
        return this->root_;
    }

    node_t *inserted_node = insert(this->root_, key);
    fix_insertion(inserted_node);

    return inserted_node;
}

void red_black_tree::transplant(node_t *node_x, node_t *node_y) {
    node_t *node_x_parent = node_x->parent;
    if (node_x_parent == nullptr) {
        this->root_ = node_y;
    } else if (node_x == node_x_parent->left) {
        node_x_parent->left = node_y;
    } else {
        node_x_parent->right = node_y;
    }
    if (node_y != nullptr) {
        node_y->parent = node_x_parent;
    }
}

void red_black_tree::fix_removal(node_t *node) {
    while (node != this->root_ and node->colour == node_t::colour_t::BLACK) {
        node_t *temp = nullptr;
        if (node->parent->left == node) {
            temp = node->parent->right;
            if (temp->colour == node_t::colour_t::RED) {
                temp->colour = node_t::colour_t::BLACK;
                node->parent->colour = node_t::colour_t::RED;
                left_rotate(node->parent);
                temp = node->parent->right;
            }
            if ((temp->right == nullptr or temp->right->colour == node_t::colour_t::BLACK) and
                (temp->left == nullptr or temp->left->colour == node_t::colour_t::BLACK)) {
				temp->colour = node_t::colour_t::RED;
				node = node->parent;
			} else {
                if (temp->right == nullptr or temp->right->colour == node_t::colour_t::BLACK) {
					temp->left->colour = node_t::colour_t::BLACK;
					temp->colour = node_t::colour_t::RED;
					right_rotate(temp);
					temp = node->parent->right;
				}
				temp->colour = node->parent->colour;
				node->parent->colour = node_t::colour_t::BLACK;
				temp->right->colour = node_t::colour_t::BLACK;
				left_rotate(node->parent);
				node = this->root_;
            }
        } else {
            temp = node->parent->left;
			if (temp->colour == node_t::colour_t::RED) {
				temp->colour = node_t::colour_t::BLACK;
				node->parent->colour = node_t::colour_t::RED;
				right_rotate(node->parent);
				temp = node->parent->left;
			}
			if ((temp->right == nullptr or temp->right->colour == node_t::colour_t::BLACK) and
                (temp->left == nullptr or temp->left->colour == node_t::colour_t::BLACK)) {
				temp->colour = node_t::colour_t::RED;
				node = node->parent;
			}
			else {
				if (temp->left == nullptr or temp->left->colour == node_t::colour_t::BLACK) {
					temp->right->colour = node_t::colour_t::BLACK;
					temp->colour = node_t::colour_t::RED;
					left_rotate(temp);
					temp = node->parent->left;
				}
				temp->colour = node->parent->colour;
				node->parent->colour = node_t::colour_t::BLACK;
				temp->left->colour = node_t::colour_t::BLACK;
				right_rotate(node->parent);
				node = this->root_;
			}
        }
    }
    if (node != nullptr) {
        node->colour = node_t::colour_t::BLACK;
    }
}

void red_black_tree::remove(node_t *node_to_remove) {
    node_t *replaced = nullptr;
    node_t *node = node_to_remove;
    node_t::colour_t node_original_colour = node->colour;
    if (node_to_remove->left == nullptr) {
        replaced = node_to_remove->right;
        transplant(node_to_remove, node_to_remove->right);
    } else if (node_to_remove->right == nullptr) {
        replaced = node_to_remove->left;
        transplant(node_to_remove, node_to_remove->left);
    } else {
        node = successor(node_to_remove);
        node_original_colour = node->colour;
        replaced = node->right;
        if (node->parent == node_to_remove) {
            replaced->parent = node;
        } else {
            transplant(node, node->right);
            node->right = node_to_remove->right;
            if (node->right != nullptr) {
                node->right->parent = node;
            }
        }
        transplant(node_to_remove, node);
        node->left = node_to_remove->left;
        if (node->left != nullptr) {
            node->left->parent = node;
        }
        node->colour = node_to_remove->colour;
    }

    if (node_original_colour == node_t::colour_t::BLACK) {
        fix_removal(replaced);
    }
}

}  // namespace tt_metal

}  // namespace tt
