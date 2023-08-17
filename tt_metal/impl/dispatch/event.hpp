#pragma once

#include <future>

namespace tt::tt_metal{
    class CommandQueue;

    class Event {
        Event() = delete;
        Event( const CommandQueue & cq, std::future<void> f ) : fut_{f}, cq_(cq) {}

        void wait() const{
            for ( const auto & f : fut_)
                f.wait();
        }


        void addToWaitList ( const Event & e ){
            fut_.insert(fut_.end(), e.fut_.begin(), e.fut_.end());
        }
        const CommandQueue& cq() const{ return cq_; }
    private:
        std::vector< std::future<void> > fut_;
        const CommandQueue & cq_;
    };


}
